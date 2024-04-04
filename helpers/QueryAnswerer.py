import logging

from typing import Optional
from fastapi import UploadFile
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.schema import HumanMessage
from langchain.callbacks.manager import CallbackManager
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain_core.callbacks.base import Callbacks

from helpers.tools import youtube_search
from helpers.StorageManager import store_manager
from utils.ImageProcessing import convert_image_to_base64

logger = logging.getLogger(__name__)

PREFIX = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""

SUFFIX = """Begin!"
{chat_history}
Question: {input}
{agent_scratchpad}"""

IMAGE_EXTRACTION_PROMPT = """Extract all 
1. questions 
2. relevant details 
3. topics 
from the image if any.
"""


class QueryAnswerer:
    def __init__(self, stream_callback: Callbacks):
        self.llm = ChatOpenAI(
            model_name="gpt-4-vision-preview",
            temperature=1.0,
            streaming=True,
            callbacks=stream_callback,
            verbose=True,
        )

        self.tools = [youtube_search]

    async def answer_query(
        self, conversation_id: str, query: Optional[str], image: Optional[UploadFile]
    ):
        if not query and not image:
            raise ValueError('At least one of "query" or "image" must be provided.')

        question = await self.process_query_image(query, image)

        logger.info(f"Systhesized question: {question}")

        prompt = ZeroShotAgent.create_prompt(
            self.tools,
            prefix=PREFIX,
            suffix=SUFFIX,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )
        memory = store_manager.get_memory(conversation_id=conversation_id)

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=self.tools, verbose=True)
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            memory=memory,
        )

        answer = await agent_chain.arun(input=question)
        store_manager.store(conversation_id=conversation_id, memory=memory)
        return answer

    async def process_query_image(
        self, query: Optional[str], image: Optional[UploadFile]
    ) -> str:
        question = ""
        if query:
            question = f"{query}\n"
        if image:
            base64_image = await convert_image_to_base64(image=image)
            image_details = self.image_summarize(
                img_base64=base64_image, prompt=IMAGE_EXTRACTION_PROMPT
            )
            question += f"{image_details}\n"
        return question

    def image_summarize(self, img_base64, prompt):
        """Make image summary"""
        chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)

        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                        },
                    ]
                )
            ]
        )
        return msg.content
