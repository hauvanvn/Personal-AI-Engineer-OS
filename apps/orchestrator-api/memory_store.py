from typing import List, Optional, Any
import psycopg
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage


def extract_clean_text(content: Any) -> str:
    """Safely extract plain text from either a string or Gemini's list-of-blocks format."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return ""


class ExtractedFact(BaseModel):
    category: str = Field(
        ...,
        description="Category of the fact: 'tech_stack', 'preference', 'project', or 'workflow'.",
    )
    fact_content: str = Field(
        ...,
        description="Concise, standalone third-person statement about the user.",
    )


class ExtractionResult(BaseModel):
    should_save: bool = Field(
        ...,
        description="True ONLY if the conversation contains new, long-term personal/technical facts.",
    )
    facts: List[ExtractedFact] = Field(default_factory=list)


class LongTermMemoryManager:
    def __init__(self, db_uri: str, api_key: str):
        self.db_uri = db_uri
        self.extractor_llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=api_key,
            # temperature=0.0,
            # transport="rest",
            timeout=15.0,
        ).with_structured_output(ExtractionResult)
        self._init_table()

    def _init_table(self):
        """Ensure the long-term user memories table exists in PostgreSQL."""
        with psycopg.connect(self.db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_memories (
                        id SERIAL PRIMARY KEY,
                        category VARCHAR(50) NOT NULL,
                        fact_content TEXT UNIQUE NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
            conn.commit()

    def get_user_profile_context(self, limit: int = 15) -> str:
        """Fetch top user facts to inject into the orchestrator system prompt."""
        with psycopg.connect(self.db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT category, fact_content FROM user_memories "
                    "ORDER BY updated_at DESC LIMIT %s",
                    (limit,),
                )
                rows = cur.fetchall()

        if not rows:
            return "No prior user profile facts recorded yet."

        return "\n".join([f"- [{r[0].upper()}] {r[1]}" for r in rows])

    async def extract_and_store_async(self, all_messages: List[BaseMessage]):
        """Run asynchronously in the background to extract new facts across tool-calling turns."""
        try:
            # Filter only Human and AI messages that have actual text content (ignoring raw ToolMessages)
            meaningful_lines = []
            for m in all_messages[-6:]:  # Inspect last 6 messages to cover multi-step tool calls
                if m.type in ("human", "ai"):
                    text = extract_clean_text(m.content)
                    if text:
                        meaningful_lines.append(f"{m.type.upper()}: {text}")

            dialogue_text = "\n".join(meaningful_lines)
            if not dialogue_text.strip():
                return

            existing_profile = self.get_user_profile_context()
            prompt = (
                "Analyze the recent dialogue and extract any NEW permanent technical preferences, "
                "operating systems, programming languages, or project contexts about the user.\n"
                "Do NOT extract temporary tasks or facts already present in the Existing Profile.\n\n"
                f"Existing Profile:\n{existing_profile}\n\n"
                f"Recent Dialogue:\n{dialogue_text}"
            )

            result: Optional[ExtractionResult] = await self.extractor_llm.ainvoke(
                [
                    SystemMessage(content="You are a precise Long-Term Memory Extractor."),
                    HumanMessage(content=prompt),
                ]
            )

            if result and result.should_save and result.facts:
                with psycopg.connect(self.db_uri) as conn:
                    with conn.cursor() as cur:
                        for item in result.facts:
                            cur.execute(
                                """
                                INSERT INTO user_memories (category, fact_content, updated_at)
                                VALUES (%s, %s, CURRENT_TIMESTAMP)
                                ON CONFLICT (fact_content) DO UPDATE
                                SET updated_at = CURRENT_TIMESTAMP;
                                """,
                                (item.category, item.fact_content),
                            )
                    conn.commit()
                print(
                    f"  [Background Memory] Stored {len(result.facts)} new user fact(s): "
                    f"{[f.fact_content for f in result.facts]}"
                )
            else:
                print("  [Background Memory] No new permanent facts detected in this turn.")
        except Exception as e:
            print(f"  [Background Memory Error] Extraction failed: {str(e)}")