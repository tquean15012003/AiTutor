from langchain_community.tools import YouTubeSearchTool

youtube_search = YouTubeSearchTool(
    name="Youtube Search",
    description="When the user poses an educational query, such as a question related to mathematics, the system should initially offer recommendations for relevant video tutorials. These suggestions are intended to enable the user to explore the topic further on their own.",
)
