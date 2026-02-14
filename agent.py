"""
Turkish AI Agent with LangGraph
Multi-agent system for Wikipedia search, mathematical calculations, and conversational AI.
"""

import json
import re
import textwrap
from typing import Dict, Any, Optional

import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph


# Model configuration
MODEL_ID = "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"
USE_4BIT_QUANTIZATION = True


class ModelLoader:
    """Handles model and tokenizer loading with optional 4-bit quantization."""
    
    @staticmethod
    def load(model_id: str = MODEL_ID, use_4bit: bool = USE_4BIT_QUANTIZATION):
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
            )
        else:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=dtype,
            )
        
        model.eval()
        return tokenizer, model


class LLMEngine:
    """Handles language model inference."""
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
    
    def _get_terminators(self):
        terms = [self.tokenizer.eos_token_id]
        eot_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(eot_id, int) and eot_id != self.tokenizer.unk_token_id:
            terms.append(eot_id)
        return terms
    
    @torch.inference_mode()
    def generate(self, messages: list, max_new_tokens: int = 256) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=self._get_terminators(),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    def chat(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 256) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.generate(messages, max_new_tokens)


class WikipediaSearcher:
    """Wikipedia API wrapper for Turkish language searches."""
    
    API_URL = "https://tr.wikipedia.org/w/api.php"
    TIMEOUT = 15
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search(self, query: str) -> Optional[Dict[str, str]]:
        """
        Search Wikipedia and return article summary.
        
        Args:
            query: Search term
            
        Returns:
            Dict with title, summary, and url, or None if not found
        """
        try:
            cleaned_query = self._clean_query(query)
            print(f"🔍 Wikipedia search: '{cleaned_query}'")
            
            title = self._search_title(cleaned_query)
            if not title:
                print(f"❌ No results found: {cleaned_query}")
                return None
            
            print(f"✓ Found title: {title}")
            return self._fetch_summary(title)
            
        except Exception as e:
            print(f"❌ Wikipedia error: {e}")
            return None
    
    def _clean_query(self, query: str) -> str:
        """Remove Turkish question suffixes from query."""
        cleaned = re.sub(r"\b(kimdir|nedir|hakkında)\b", "", query, flags=re.IGNORECASE)
        return cleaned.strip() if cleaned.strip() else query
    
    def _search_title(self, query: str) -> Optional[str]:
        """Search for article title."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
        }
        
        response = self.session.get(self.API_URL, params=params, timeout=self.TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("query", {}).get("search", [])
        return results[0]["title"] if results else None
    
    def _fetch_summary(self, title: str) -> Optional[Dict[str, str]]:
        """Fetch article summary by title."""
        params = {
            "action": "query",
            "prop": "extracts|info",
            "exintro": True,
            "explaintext": True,
            "titles": title,
            "format": "json",
            "inprop": "url",
        }
        
        response = self.session.get(self.API_URL, params=params, timeout=self.TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        page = list(pages.values())[0]
        
        extract = page.get("extract", "")
        url = page.get("fullurl", "")
        
        if not extract:
            print(f"❌ No summary found: {title}")
            return None
        
        summary = extract[:800] + ("..." if len(extract) > 800 else "")
        print(f"✓ Summary retrieved: {len(summary)} characters")
        
        return {
            "title": title,
            "summary": summary,
            "url": url
        }


class AgentOrchestrator:
    """Main agent system using LangGraph for workflow orchestration."""
    
    ROUTER_PROMPT = """\
Analyze the user's message and respond with JSON indicating the appropriate action.

RULES:
- Greeting/thanks/casual chat → {"action":"final"}
- Math calculation → {"action":"math","query":"expression"}
- Information/research query → {"action":"wiki","query":"search term"}

EXAMPLES:
hello → {"action":"final"}
thank you → {"action":"final"}
2+2 → {"action":"math","query":"2+2"}
(12*5)-7 → {"action":"math","query":"(12*5)-7"}
sqrt(16) → {"action":"math","query":"sqrt(16)"}
Who is Atatürk → {"action":"wiki","query":"Atatürk"}
What is Python → {"action":"wiki","query":"Python"}

Output ONLY valid JSON, nothing else.
"""
    
    ANSWER_PROMPT = """\
You are a helpful assistant. Answer the user's question based on the Wikipedia summary provided.

RULES:
- Use the Wikipedia summary if available
- If no summary, explain this politely
- Keep answer to 5-10 sentences
- Be clear, concise, and accurate
- Do not fabricate information
- Respond in Turkish
"""
    
    FINAL_PROMPT = """\
You are a friendly and helpful assistant. Have a natural conversation with the user.

RULES:
- Keep responses brief and to the point
- Be natural and friendly
- Respond in Turkish
"""
    
    MATH_PROMPT = """\
You are a mathematics assistant.

RULES:
- Only perform mathematical calculations
- Provide the exact result
- Do not add unnecessary commentary
- If the expression is unclear, ask for clarification
- Respond in Turkish
"""
    
    def __init__(self, llm_engine: LLMEngine, wiki_searcher: WikipediaSearcher):
        self.llm = llm_engine
        self.wiki = wiki_searcher
        self.app = self._build_graph()
    
    def _build_graph(self) -> CompiledStateGraph:
        """Construct the LangGraph workflow."""
        graph = StateGraph(dict)
        
        graph.add_node("router", self._router_node)
        graph.add_node("wiki", self._wiki_node)
        graph.add_node("answer", self._answer_node)
        graph.add_node("final", self._final_node)
        graph.add_node("math", self._math_node)
        
        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router", 
            self._route_decision,
            {"wiki": "wiki", "final": "final", "math": "math"}
        )
        graph.add_edge("wiki", "answer")
        graph.add_edge("answer", END)
        graph.add_edge("final", END)
        graph.add_edge("math", END)
        
        return graph.compile()
    
    def _router_node(self, state: dict) -> dict:
        """Route user query to appropriate handler."""
        user_query = state["user_query"].strip()
        
        response = self.llm.chat(
            system_prompt=self.ROUTER_PROMPT,
            user_prompt=user_query,
            max_new_tokens=50
        )
        
        print(f"🤖 LLM routing: {response}")
        
        # Parse JSON from response
        match = re.search(r'\{.*?\}', response, flags=re.DOTALL)
        if match:
            try:
                plan = json.loads(match.group(0))
                if "query" not in plan:
                    plan["query"] = ""
                if plan.get("action") not in ["wiki", "final", "math"]:
                    plan = {"action": "final", "query": ""}
            except json.JSONDecodeError:
                plan = {"action": "final", "query": ""}
        else:
            plan = {"action": "final", "query": ""}
        
        action_label = {
            "wiki": "📚 WIKI",
            "math": "🧮 MATH",
            "final": "💬 CHAT"
        }.get(plan["action"], "❌ ERROR")
        
        print(f"📋 Plan: {action_label} → {plan}")
        
        return {**state, "plan": plan}
    
    def _wiki_node(self, state: dict) -> dict:
        """Search Wikipedia for information."""
        query = state.get("plan", {}).get("query") or state["user_query"]
        
        try:
            result = self.wiki.search(query)
            if result:
                print(f"✓ Wikipedia found: {result['title']}")
            else:
                print("❌ Wikipedia not found")
        except Exception as e:
            print(f"❌ Wiki error: {e}")
            result = None
        
        return {**state, "wiki": result}
    
    def _answer_node(self, state: dict) -> dict:
        """Generate answer based on Wikipedia results."""
        user_query = state["user_query"]
        wiki_data = state.get("wiki")
        
        if wiki_data:
            user_prompt = f"""Question: {user_query}

Wikipedia Information:
Title: {wiki_data['title']}
Summary: {wiki_data['summary']}
URL: {wiki_data['url']}

Answer the question based on the Wikipedia information above."""
        else:
            user_prompt = f"""Question: {user_query}

No Wikipedia information found. Politely explain this and help as best you can."""
        
        answer = self.llm.chat(
            system_prompt=self.ANSWER_PROMPT,
            user_prompt=user_prompt,
            max_new_tokens=320
        )
        
        return {**state, "answer": answer}
    
    def _final_node(self, state: dict) -> dict:
        """Handle casual conversation."""
        answer = self.llm.chat(
            system_prompt=self.FINAL_PROMPT,
            user_prompt=state["user_query"],
            max_new_tokens=200
        )
        return {**state, "answer": answer}
    
    def _math_node(self, state: dict) -> dict:
        """Handle mathematical calculations."""
        expression = state.get("plan", {}).get("query") or state["user_query"]
        
        user_prompt = f"""Calculate the expression and provide the result:
{expression}

Output format:
Result: <value>
"""
        answer = self.llm.chat(
            system_prompt=self.MATH_PROMPT,
            user_prompt=user_prompt,
            max_new_tokens=120
        )
        
        return {**state, "answer": answer}
    
    def _route_decision(self, state: dict) -> str:
        """Determine next node based on router decision."""
        action = state.get("plan", {}).get("action", "final")
        return action
    
    def process_query(self, query: str) -> dict:
        """Process a user query through the agent system."""
        initial_state = {
            "user_query": query,
            "plan": None,
            "wiki": None,
            "answer": None
        }
        return self.app.invoke(initial_state)


def run_interactive_mode():
    """Run the agent in interactive mode."""
    print("🚀 Loading system...")
    
    # Initialize components
    tokenizer, model = ModelLoader.load()
    llm_engine = LLMEngine(tokenizer, model)
    wiki_searcher = WikipediaSearcher()
    agent = AgentOrchestrator(llm_engine, wiki_searcher)
    
    print("\n" + "="*60)
    print("INTERACTIVE MODE STARTED")
    print("="*60)
    print("Ask your questions (press Enter with empty input to exit)\n")
    
    while True:
        user_input = input("🧑 Question: ").strip()
        if not user_input:
            print("\n👋 Goodbye!")
            break
        
        print("\n" + "─"*60)
        result = agent.process_query(user_input)
        
        print("\n🤖 ANSWER:")
        print("─"*60)
        print(textwrap.fill(result["answer"], width=100))
        
        if result.get("wiki"):
            print("\n📚 WIKIPEDIA SOURCE:")
            print("─"*60)
            print(f"Title: {result['wiki']['title']}")
            print(f"URL: {result['wiki']['url']}")
        
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    run_interactive_mode()

