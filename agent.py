"""
Industrial-Grade Multi-Agent System with LangGraph
===================================================
Agents:
  - PlannerAgent       : Görev analizi ve çok adımlı plan üretimi
  - RouterAgent        : Plana göre iş akışı yönlendirme
  - WikiSearchAgent    : Türkçe Wikipedia araştırma
  - SummarizerAgent    : Uzun içerikleri yoğunlaştırma
  - MathAgent          : Matematiksel hesaplamalar
  - ChatAgent          : Sohbet / genel konuşma
  - AnswerGeneratorAgent: Nihai cevap üretimi
  - QualityCheckerAgent: Kalite skoru + otomatik yeniden deneme
  - ConversationMemory : Konuşma geçmişi yönetimi
  - SystemLogger       : Yapılandırılmış log + yürütme izi
"""

from __future__ import annotations

import json
import re
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

import requests
import torch
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ──────────────────────────────────────────────
# Konfigurasyon
# ──────────────────────────────────────────────
MODEL_ID = "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"
USE_4BIT_QUANTIZATION = True
MAX_RETRY_ATTEMPTS = 2
QUALITY_THRESHOLD = 6       # 0-10 skalasında minimum kalite skoru
MEMORY_WINDOW = 6           # Hafızada tutulacak maksimum tur sayısı
WIKI_SUMMARY_MAX_CHARS = 1200
WIKI_CONDENSED_MAX_CHARS = 500


# ──────────────────────────────────────────────
# Durum (State) Tanımı
# ──────────────────────────────────────────────
class AgentState(TypedDict):
    """LangGraph boyunca taşınan merkezi durum nesnesi."""
    session_id: str
    user_query: str
    conversation_history: List[Dict[str, str]]

    # Planner çıktısı
    plan: Optional[Dict[str, Any]]

    # Araç çıktıları
    wiki_raw: Optional[Dict[str, str]]
    wiki_summary: Optional[str]
    math_result: Optional[str]

    # Cevap üretimi
    draft_answer: Optional[str]
    final_answer: Optional[str]

    # Kalite kontrol
    qa_report: Optional[Dict[str, Any]]
    retry_count: int

    # İzleme
    execution_trace: List[str]
    error_log: List[str]


# ──────────────────────────────────────────────
# Sistem Logger
# ──────────────────────────────────────────────
class SystemLogger:
    """Renkli, yapılandırılmış konsol logu."""

    ICONS = {
        "planner":   "🗺️ ",
        "router":    "🔀",
        "wiki":      "📚",
        "summarizer":"📝",
        "math":      "🧮",
        "chat":      "💬",
        "answer":    "✍️ ",
        "qa":        "🔍",
        "memory":    "🧠",
        "system":    "⚙️ ",
        "ok":        "✅",
        "warn":      "⚠️ ",
        "error":     "❌",
        "retry":     "🔄",
    }

    @staticmethod
    def log(agent: str, message: str) -> str:
        icon = SystemLogger.ICONS.get(agent, "•")
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {icon} [{agent.upper():12s}] {message}"
        print(line)
        return line

    @staticmethod
    def separator(title: str = "", width: int = 70):
        if title:
            pad = (width - len(title) - 2) // 2
            print("─" * pad + f" {title} " + "─" * pad)
        else:
            print("─" * width)

    @staticmethod
    def header(title: str, width: int = 70):
        print("\n" + "═" * width)
        print(f"  {title}")
        print("═" * width)


# ──────────────────────────────────────────────
# Konuşma Hafızası
# ──────────────────────────────────────────────
class ConversationMemory:
    """Kayan pencereli konuşma geçmişi yöneticisi."""

    def __init__(self, window: int = MEMORY_WINDOW):
        self.window = window
        self._turns: List[Dict[str, str]] = []

    def add(self, role: str, content: str):
        self._turns.append({"role": role, "content": content})
        if len(self._turns) > self.window * 2:
            self._turns = self._turns[-(self.window * 2):]

    def get_context(self) -> str:
        """Son N turu metin olarak döndür."""
        if not self._turns:
            return "Henüz konuşma geçmişi yok."
        lines = []
        for t in self._turns[-self.window * 2:]:
            prefix = "Kullanıcı" if t["role"] == "user" else "Asistan"
            lines.append(f"{prefix}: {t['content']}")
        return "\n".join(lines)

    def as_list(self) -> List[Dict[str, str]]:
        return list(self._turns)


# ──────────────────────────────────────────────
# Model Yükleyici
# ──────────────────────────────────────────────
class ModelLoader:
    @staticmethod
    def load(model_id: str = MODEL_ID, use_4bit: bool = USE_4BIT_QUANTIZATION):
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        if use_4bit:
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=qcfg,
                torch_dtype=torch.float16,
            )
        else:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype=dtype
            )

        model.eval()
        return tokenizer, model


# ──────────────────────────────────────────────
# LLM Motoru
# ──────────────────────────────────────────────
class LLMEngine:
    """Dil modeli çıkarım motoru."""

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def _terminators(self):
        terms = [self.tokenizer.eos_token_id]
        eot = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(eot, int) and eot != self.tokenizer.unk_token_id:
            terms.append(eot)
        return terms

    @torch.inference_mode()
    def generate(self, messages: List[Dict], max_new_tokens: int = 256) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=self._terminators(),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def chat(self, system_prompt: str, user_prompt: str,
             max_new_tokens: int = 256, history: Optional[List[Dict]] = None) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        return self.generate(messages, max_new_tokens)


# ──────────────────────────────────────────────
# Wikipedia Arayıcı
# ──────────────────────────────────────────────
class WikipediaSearcher:
    API_URL = "https://tr.wikipedia.org/w/api.php"
    TIMEOUT = 15

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "IndustrialAgentSystem/2.0 (LangGraph; Turkish NLP)"
        })

    def search(self, query: str) -> Optional[Dict[str, str]]:
        try:
            cleaned = re.sub(r"\b(kimdir|nedir|hakkında|ne zaman|nerede)\b",
                             "", query, flags=re.IGNORECASE).strip() or query
            SystemLogger.log("wiki", f"Aranıyor: '{cleaned}'")

            title = self._find_title(cleaned)
            if not title:
                return None

            return self._fetch(title)
        except Exception as e:
            SystemLogger.log("error", f"Wikipedia hatası: {e}")
            return None

    def _find_title(self, query: str) -> Optional[str]:
        params = {
            "action": "query", "list": "search",
            "srsearch": query, "format": "json", "srlimit": 1
        }
        r = self.session.get(self.API_URL, params=params, timeout=self.TIMEOUT)
        r.raise_for_status()
        results = r.json().get("query", {}).get("search", [])
        return results[0]["title"] if results else None

    def _fetch(self, title: str) -> Optional[Dict[str, str]]:
        params = {
            "action": "query", "prop": "extracts|info",
            "exintro": True, "explaintext": True,
            "titles": title, "format": "json", "inprop": "url"
        }
        r = self.session.get(self.API_URL, params=params, timeout=self.TIMEOUT)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        page = list(pages.values())[0]
        extract = page.get("extract", "")
        if not extract:
            return None
        summary = extract[:WIKI_SUMMARY_MAX_CHARS] + ("…" if len(extract) > WIKI_SUMMARY_MAX_CHARS else "")
        return {
            "title": title,
            "summary": summary,
            "url": page.get("fullurl", ""),
            "char_count": str(len(extract))
        }


# ──────────────────────────────────────────────
# Ajan İstem Kitaplığı
# ──────────────────────────────────────────────
class Prompts:
    PLANNER = """\
Sen bir görev planlama ajanısın. Kullanıcının sorgusunu analiz et ve JSON planı üret.

PLAN ŞEMASI:
{
  "task_type": "<research|math|chat>",
  "steps": ["<adım1>", "<adım2>", ...],
  "search_query": "<Wikipedia arama terimi veya boş>",
  "requires_summary": <true|false>,
  "complexity": "<low|medium|high>",
  "language": "tr",
  "intent": "<kullanıcı niyetinin kısa açıklaması>"
}

KURALLAR:
- Araştırma/bilgi sorusu → task_type: "research", steps: ["search_wiki","summarize","generate_answer"]
- Matematik → task_type: "math", steps: ["calculate","generate_answer"]
- Selamlama/sohbet → task_type: "chat", steps: ["generate_answer"]
- requires_summary: Wikipedia sonucu uzunsa (complexity medium/high) true
- Yalnızca JSON çıktısı ver.
"""

    ROUTER = """\
Plan doğrultusunda yönlendir. Sadece tek kelimeyle yanıt ver:
  research  → wiki araması gerekiyor
  math      → matematik hesabı gerekiyor
  chat      → doğrudan sohbet
"""

    SUMMARIZER = """\
Sen bir içerik özetleme uzmanısın. Verilen Wikipedia metnini sıkıştır.

KURALLAR:
- En önemli 4-6 bilgiyi madde madde listele
- Her madde 1-2 cümle
- Anahtar kavramları, tarihleri, isimleri koru
- Türkçe yaz
- Gereksiz detayları at
"""

    ANSWER_GENERATOR = """\
Sen uzman bir asistansın. Plandan ve toplanan verilerden yararlanarak kullanıcının sorusunu yanıtla.

KURALLAR:
- Özetlenmiş bilgiyi kullan, ham veri yerine
- 5-8 cümle, akıcı paragraf
- Kaynak belirt (Wikipedia varsa)
- Türkçe, açık ve doğru yaz
- Bilmediğin şeyi uydurma
"""

    MATH = """\
Sen bir matematik asistanısın.

KURALLAR:
- Verilen ifadeyi hesapla
- Adım adım çözüm göster
- Sonucu "Sonuç: <değer>" formatında sun
- Türkçe yaz
"""

    CHAT = """\
Sen samimi ve yardımsever bir asistansın. Kullanıcıyla doğal Türkçe konuşma yap.

KURALLAR:
- Kısa ve içten yanıt ver
- Konuşma geçmişini dikkate al
- Türkçe yaz
"""

    QA_CHECKER = """\
Sen bir kalite kontrol ajanısın. Üretilen cevabı değerlendir ve JSON raporu ver.

RAPOR ŞEMASI:
{
  "quality_score": <0-10 tam sayı>,
  "issues": ["<sorun1>", ...],
  "suggestions": ["<öneri1>", ...],
  "approved": <true|false>,
  "reasoning": "<kısa değerlendirme>"
}

DEĞERLENDİRME KRİTERLERİ:
- Soruyla ilgililik (0-3)
- Doğruluk ve tutarlılık (0-3)
- Anlaşılırlık ve akıcılık (0-2)
- Yanıltıcı veya boş içerik cezası (-2)

approved = true ise quality_score >= 6 olmalı.
Yalnızca JSON çıktısı ver.
"""


# ──────────────────────────────────────────────
# Ajan Uygulamaları
# ──────────────────────────────────────────────
class PlannerAgent:
    """Gelen sorguyu analiz eder ve çok adımlı yürütme planı üretir."""

    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        trace = state["execution_trace"]
        query = state["user_query"]
        history_ctx = ""
        if state["conversation_history"]:
            last = state["conversation_history"][-4:]
            history_ctx = "\n".join(
                f"{'K' if t['role']=='user' else 'A'}: {t['content']}" for t in last
            )

        user_prompt = f"Geçmiş (son turlar):\n{history_ctx}\n\nMevcut sorgu: {query}"
        raw = self.llm.chat(Prompts.PLANNER, user_prompt, max_new_tokens=200)
        SystemLogger.log("planner", f"Ham plan çıktısı: {raw[:120]}…")

        plan = self._parse_plan(raw)
        trace.append(SystemLogger.log("planner", f"Plan → {plan}"))
        return {**state, "plan": plan, "execution_trace": trace}

    @staticmethod
    def _parse_plan(raw: str) -> Dict[str, Any]:
        default = {
            "task_type": "chat",
            "steps": ["generate_answer"],
            "search_query": "",
            "requires_summary": False,
            "complexity": "low",
            "language": "tr",
            "intent": "genel sohbet"
        }
        match = re.search(r'\{.*?\}', raw, flags=re.DOTALL)
        if not match:
            return default
        try:
            plan = json.loads(match.group(0))
            for k, v in default.items():
                plan.setdefault(k, v)
            if plan["task_type"] not in ("research", "math", "chat"):
                plan["task_type"] = "chat"
            return plan
        except json.JSONDecodeError:
            return default


class WikiSearchAgent:
    """Wikipedia'da araştırma yapar."""

    def __init__(self, searcher: WikipediaSearcher):
        self.searcher = searcher

    def run(self, state: AgentState) -> AgentState:
        trace = state["execution_trace"]
        errors = state["error_log"]
        query = state["plan"].get("search_query") or state["user_query"]

        result = self.searcher.search(query)
        if result:
            trace.append(SystemLogger.log("wiki", f"Bulundu: '{result['title']}' ({result['char_count']} karakter)"))
        else:
            errors.append("Wikipedia: sonuç bulunamadı.")
            trace.append(SystemLogger.log("warn", "Wikipedia sonuç yok"))

        return {**state, "wiki_raw": result, "execution_trace": trace, "error_log": errors}


class SummarizerAgent:
    """Uzun Wikipedia içeriğini yoğunlaştırır."""

    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        trace = state["execution_trace"]
        wiki = state.get("wiki_raw")

        if not wiki:
            trace.append(SystemLogger.log("summarizer", "Özetlenecek içerik yok, atlanıyor"))
            return {**state, "wiki_summary": None, "execution_trace": trace}

        user_prompt = (
            f"Makale başlığı: {wiki['title']}\n\n"
            f"İçerik:\n{wiki['summary']}\n\n"
            f"Kullanıcı sorusu: {state['user_query']}\n\n"
            "Lütfen bu içeriği soruya odaklanarak özetle."
        )

        compressed = self.llm.chat(Prompts.SUMMARIZER, user_prompt, max_new_tokens=300)

        if len(compressed) > WIKI_CONDENSED_MAX_CHARS:
            compressed = compressed[:WIKI_CONDENSED_MAX_CHARS] + "…"

        trace.append(SystemLogger.log("summarizer",
            f"Özet üretildi: {len(compressed)} karakter (orijinal: {len(wiki['summary'])})"))

        return {**state, "wiki_summary": compressed, "execution_trace": trace}


class MathAgent:
    """Matematiksel hesaplamalar gerçekleştirir."""

    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        trace = state["execution_trace"]
        expr = state["plan"].get("search_query") or state["user_query"]

        user_prompt = f"İfade: {expr}\n\nAdım adım çöz ve sonucu ver."
        result = self.llm.chat(Prompts.MATH, user_prompt, max_new_tokens=200)

        trace.append(SystemLogger.log("math", f"Hesaplama tamamlandı"))
        return {**state, "math_result": result, "execution_trace": trace}


class AnswerGeneratorAgent:
    """Toplanan tüm verilerden nihai taslak cevap üretir."""

    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        trace = state["execution_trace"]
        plan = state["plan"]
        task_type = plan["task_type"]

        if task_type == "math":
            draft = state.get("math_result") or "Hesaplama sonucu bulunamadı."
            trace.append(SystemLogger.log("answer", "Matematik cevabı iletildi"))
            return {**state, "draft_answer": draft, "execution_trace": trace}

        if task_type == "chat":
            history = state["conversation_history"][-4:] if state["conversation_history"] else []
            draft = self.llm.chat(
                Prompts.CHAT, state["user_query"],
                max_new_tokens=200, history=history
            )
            trace.append(SystemLogger.log("answer", "Sohbet cevabı üretildi"))
            return {**state, "draft_answer": draft, "execution_trace": trace}

        # research
        summary = state.get("wiki_summary")
        raw = state.get("wiki_raw")
        wiki_info = ""
        if summary:
            wiki_info = f"Özet Bilgi:\n{summary}"
        elif raw:
            wiki_info = f"Ham Bilgi (özetsiz):\n{raw['summary'][:600]}"
        else:
            wiki_info = "Wikipedia'da ilgili bilgi bulunamadı."

        user_prompt = (
            f"Kullanıcı Sorusu: {state['user_query']}\n\n"
            f"Plan Niyeti: {plan.get('intent', '')}\n\n"
            f"{wiki_info}\n\n"
            + (f"Kaynak: {raw['url']}" if raw else "")
        )
        draft = self.llm.chat(Prompts.ANSWER_GENERATOR, user_prompt, max_new_tokens=380)
        trace.append(SystemLogger.log("answer", "Araştırma cevabı üretildi"))
        return {**state, "draft_answer": draft, "execution_trace": trace}


class QualityCheckerAgent:
    """Taslak cevabın kalitesini değerlendirir; düşük skorlarda yeniden deneme tetikler."""

    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        trace = state["execution_trace"]
        draft = state.get("draft_answer", "")
        retry = state.get("retry_count", 0)

        user_prompt = (
            f"Kullanıcı Sorusu: {state['user_query']}\n\n"
            f"Üretilen Cevap:\n{draft}\n\n"
            "Bu cevabı yukarıdaki kriterlere göre değerlendir."
        )
        raw = self.llm.chat(Prompts.QA_CHECKER, user_prompt, max_new_tokens=200)
        report = self._parse_report(raw)

        score = report.get("quality_score", 0)
        approved = report.get("approved", False)

        trace.append(SystemLogger.log("qa",
            f"Kalite skoru: {score}/10 | Onaylı: {approved} | "
            f"Sorunlar: {report.get('issues', [])}"))

        if not approved and retry < MAX_RETRY_ATTEMPTS:
            trace.append(SystemLogger.log("retry", f"Yeniden deneme #{retry + 1} tetikleniyor"))
            return {
                **state,
                "qa_report": report,
                "retry_count": retry + 1,
                "draft_answer": None,
                "execution_trace": trace,
            }

        final = draft
        if not approved:
            final = (
                f"{draft}\n\n"
                f"⚠️ Not: Bu yanıt kalite eşiğinin altında kalabilir "
                f"(skor: {score}/10). Lütfen bilgileri teyit edin."
            )
            trace.append(SystemLogger.log("warn", "Düşük kaliteli cevap uyarıyla yayımlandı"))

        return {**state, "qa_report": report, "final_answer": final, "execution_trace": trace}

    @staticmethod
    def _parse_report(raw: str) -> Dict[str, Any]:
        default = {
            "quality_score": 5,
            "issues": [],
            "suggestions": [],
            "approved": True,
            "reasoning": "Değerlendirme ayrıştırılamadı"
        }
        match = re.search(r'\{.*?\}', raw, flags=re.DOTALL)
        if not match:
            return default
        try:
            report = json.loads(match.group(0))
            for k, v in default.items():
                report.setdefault(k, v)
            return report
        except json.JSONDecodeError:
            return default


# ──────────────────────────────────────────────
# Ana Orkestratör
# ──────────────────────────────────────────────
class AgentOrchestrator:
    """
    LangGraph tabanlı endüstriyel ajan sistemi.

    Graf Akışı:
        planner → router → ┬─ wiki → summarizer ─┐
                           ├─ math               ─┤→ answer_gen → qa_checker → END
                           └─ chat              ─┘
        qa_checker → answer_gen  (kalite başarısız + retry hakkı varsa)
    """

    def __init__(
        self,
        llm: LLMEngine,
        wiki: WikipediaSearcher,
        memory: ConversationMemory,
    ):
        self.llm = llm
        self.wiki = wiki
        self.memory = memory

        self.planner   = PlannerAgent(llm)
        self.searcher  = WikiSearchAgent(wiki)
        self.summarizer= SummarizerAgent(llm)
        self.math      = MathAgent(llm)
        self.answer    = AnswerGeneratorAgent(llm)
        self.qa        = QualityCheckerAgent(llm)

        self.app = self._build_graph()

    # ── Graf inşası ────────────────────────────
    def _build_graph(self) -> CompiledStateGraph:
        g = StateGraph(AgentState)

        g.add_node("planner",     self.planner.run)
        g.add_node("router",      self._router_node)
        g.add_node("wiki",        self.searcher.run)
        g.add_node("summarizer",  self._conditional_summarizer)
        g.add_node("math",        self.math.run)
        g.add_node("chat",        self._chat_node)
        g.add_node("answer_gen",  self.answer.run)
        g.add_node("qa_checker",  self.qa.run)

        g.set_entry_point("planner")
        g.add_edge("planner", "router")

        g.add_conditional_edges(
            "router", self._route_decision,
            {"wiki": "wiki", "math": "math", "chat": "chat"}
        )

        g.add_edge("wiki",   "summarizer")
        g.add_edge("math",   "answer_gen")
        g.add_edge("chat",   "answer_gen")
        g.add_edge("summarizer", "answer_gen")

        g.add_edge("answer_gen", "qa_checker")

        g.add_conditional_edges(
            "qa_checker", self._qa_decision,
            {"approved": END, "retry": "answer_gen"}
        )

        return g.compile()

    # ── Ara Düğümler ──────────────────────────
    def _router_node(self, state: AgentState) -> AgentState:
        task = state["plan"]["task_type"]
        route_map = {"research": "wiki", "math": "math", "chat": "chat"}
        route = route_map.get(task, "chat")
        trace = state["execution_trace"]
        trace.append(SystemLogger.log("router", f"Yönlendirme: {task} → {route}"))
        return {**state, "execution_trace": trace}

    def _route_decision(self, state: AgentState) -> str:
        return {"research": "wiki", "math": "math", "chat": "chat"}.get(
            state["plan"]["task_type"], "chat"
        )

    def _conditional_summarizer(self, state: AgentState) -> AgentState:
        """Özetlemeyi plan kontrolüne göre uygula veya atla."""
        if state["plan"].get("requires_summary") and state.get("wiki_raw"):
            return self.summarizer.run(state)
        trace = state["execution_trace"]
        trace.append(SystemLogger.log("summarizer", "Özetleme atlandı (gerekmedi)"))
        return {**state, "wiki_summary": None, "execution_trace": trace}

    def _chat_node(self, state: AgentState) -> AgentState:
        """Chat için ayrı düğüm (doğrudan answer_gen'e geçiyor)."""
        return state

    def _qa_decision(self, state: AgentState) -> str:
        if state.get("final_answer"):
            return "approved"
        return "retry" if state.get("retry_count", 0) <= MAX_RETRY_ATTEMPTS else "approved"

    # ── Sorgu İşleme ─────────────────────────
    def process_query(self, query: str) -> AgentState:
        session_id = str(uuid.uuid4())[:8]
        SystemLogger.header(f"YENİ SORGU  [oturum: {session_id}]")
        SystemLogger.log("system", f"Sorgu: {query}")

        initial: AgentState = {
            "session_id": session_id,
            "user_query": query,
            "conversation_history": self.memory.as_list(),
            "plan": None,
            "wiki_raw": None,
            "wiki_summary": None,
            "math_result": None,
            "draft_answer": None,
            "final_answer": None,
            "qa_report": None,
            "retry_count": 0,
            "execution_trace": [],
            "error_log": [],
        }

        result: AgentState = self.app.invoke(initial)

        # Hafızayı güncelle
        self.memory.add("user", query)
        self.memory.add("assistant", result.get("final_answer", ""))
        SystemLogger.log("memory", f"Hafıza güncellendi ({len(self.memory.as_list())} tur)")

        return result


# ──────────────────────────────────────────────
# Sonuç Yazdırıcı
# ──────────────────────────────────────────────
def print_result(result: AgentState):
    """İşlenmiş sonucu formatlı şekilde ekrana yaz."""
    SystemLogger.separator("CEVAP")
    print(textwrap.fill(result.get("final_answer", "Cevap üretilemedi."), width=90))

    # Wikipedia kaynağı
    wiki = result.get("wiki_raw")
    if wiki:
        SystemLogger.separator("KAYNAK")
        print(f"  Başlık : {wiki['title']}")
        print(f"  URL    : {wiki['url']}")

    # QA Raporu
    qa = result.get("qa_report")
    if qa:
        SystemLogger.separator("KALİTE RAPORU")
        score = qa.get("quality_score", "?")
        approved = "✅ Onaylı" if qa.get("approved") else "⚠️ Onaysız"
        print(f"  Skor   : {score}/10  |  Durum: {approved}")
        if qa.get("issues"):
            print(f"  Sorunlar: {', '.join(qa['issues'])}")
        if qa.get("reasoning"):
            print(f"  Gerekçe: {qa['reasoning']}")

    # Yürütme özeti
    trace = result.get("execution_trace", [])
    SystemLogger.separator("YÜRÜTME İZİ")
    for i, step in enumerate(trace, 1):
        print(f"  {i:2d}. {step}")

    errors = result.get("error_log", [])
    if errors:
        SystemLogger.separator("HATALAR")
        for e in errors:
            print(f"  ⚠️  {e}")

    print("═" * 70 + "\n")


# ──────────────────────────────────────────────
# Etkileşimli Çalışma Modu
# ──────────────────────────────────────────────
def run_interactive():
    SystemLogger.header("ENDÜSTRİYEL ÇOKLU-AJAN SİSTEMİ  v2.0")
    print("  Bileşenler: Planner · Wiki · Summarizer · Math · Chat · QA Checker")
    print("  Model     :", MODEL_ID)
    print()

    SystemLogger.log("system", "Model yükleniyor…")
    tokenizer, model = ModelLoader.load()
    llm_engine   = LLMEngine(tokenizer, model)
    wiki_searcher = WikipediaSearcher()
    memory       = ConversationMemory(window=MEMORY_WINDOW)
    orchestrator = AgentOrchestrator(llm_engine, wiki_searcher, memory)

    SystemLogger.log("ok", "Sistem hazır. Sorunuzu yazın (çıkmak için boş Enter).\n")

    while True:
        try:
            user_input = input("🧑  Sorgu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            SystemLogger.log("system", "Oturum kapatıldı. Güle güle!")
            break

        result = orchestrator.process_query(user_input)
        print_result(result)


if __name__ == "__main__":
    run_interactive()
