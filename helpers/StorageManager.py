from langchain.memory import ConversationBufferMemory

class StoreManager():
    def __init__(self):
        self.storage = {}
    
    def store(self, conversation_id: str, memory: ConversationBufferMemory):
        self.storage[conversation_id] = memory

    def get_memory(self, conversation_id) -> ConversationBufferMemory:
        return self.storage.get(conversation_id, ConversationBufferMemory(memory_key="chat_history"))

store_manager = StoreManager()