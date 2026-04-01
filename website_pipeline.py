import os 
import asyncio
import tiktoken
from pathlib import Path 
from typing import Optional
from research_config import *
from dotenv import load_dotenv
from file_summarizer import FileSummarizer
from domain_selector import DomainController
from website_url_converter import URLConverter
from domain_finder import DomainResearchAssistant
from lateral_searcher.description_generator import genearate_search_query
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer


def get_valid_file_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    return path

class WebsiteController:
    def __init__(self, file_path: Optional[str]=None,url:Optional[str]=None, api_key:str=None,research_type="study", more_urls:str="normal", summarize: bool = True, find_domains:bool =False):
        self.url = url
        self.path = get_valid_file_path(file_path) if file_path else None
        self.api_key = api_key
        self.more_urls = more_urls 
        self.research_type = research_type
        self.find_domains = find_domains
        self.summarize = summarize
        self.openai_tokenizer =  OpenAITokenizer(
                tokenizer=tiktoken.encoding_for_model("gpt-4o"),    
            max_tokens=600,  # context window length required for OpenAI tokenizers
            )
 
    async def website_pipeline(self,):
        if self.url is None and self.path is None and self.query is None:
            print("No input provided") #if no input, return empty string
            return ""
        if not self.api_key:
            print("No api_key provided")
            return "" #returns nothing if no api_key provided
        if self.url:
            if not self.url.startswith(("http://", "https://")): #basic validation for url input
                print("Invalid URL format. URL should start with http:// or https://")
                return ""
            else:
                try:
                    url_converter = URLConverter(
                        openai_tokenizer=self.openai_tokenizer, 
                        url=self.url
                    )
                    url_file_path, md_text = await url_converter.url_converter() #gets the list of markdown lines and url_path for summarization

                except Exception as e:
                    print(f"Failed to convert url: {e}")
                    import traceback
                    traceback.print_exc()
                    return ""
                else:
                    try:
                        #if summarize is true and url input is valid, run summarizer on url input -> only runs if file input is invalid or not provided, prioritizes url input for summarization
                        if self.summarize and isinstance(self.url, str):
                            print("About to start summarizer")
                            web_summarizer = FileSummarizer(
                                        file_path=str(url_file_path),
                                        api_key=self.api_key,
                                        research_type=self.research_type,
                                        mode=self.more_urls,
                                    )
                                    
                            web_summary = await web_summarizer.run_summarizer()
                    except Exception as e:
                            print(f"Failed to summarize url content: {e}")
                            return ""
                    else:
                        if self.find_domains: #finds related domains given a url 
                            query = genearate_search_query(md_text[0]) #takes the md str -> llm -> query for domain research
                            domain_contorller = DomainController(
                                        api_key=self.api_key,
                                        url=self.url,
                                        query=query,
                                        more_urls=self.more_urls
                                    )
                            domain = domain_contorller.domain_extractor() #either a dict or str: dependent on mode inputted
                            if self.more_urls == "normal" and isinstance(domain, str):
                                        assistant = DomainResearchAssistant(
                                            query=query,
                                            domain=domain,
                                            api_key=self.api_key
                                        )

                                        config = ResearchConfig(domain=domain)
                                        domain_results = await assistant.research_crawler(config)
                            if self.more_urls == "expand" and isinstance(domain, dict):
                                        assistant = DomainResearchAssistant(
                                        query=query,
                                        domain=None,
                                        api_key=self.api_key
                                        )
                                        domain_results = await assistant.multi_domain_research(domain)
                            return web_summary, domain_results  
                        return web_summary
        #if summarize is true and path is valid, run summarizer on file input -> prioritizes file input over url input for summarization
        if self.summarize and isinstance(self.path, Path): 
            file_summarizer = FileSummarizer(
                                file_path=self.path,
                                api_key=self.api_key,
                                research_type=self.research_type,
                                mode=self.more_urls,
                        )
                            
            return await file_summarizer.run_summarizer()                  
        #write file to markdown
        
if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    website_controller = WebsiteController(
        url="https://www.ibm.com/think/topics/cyber-hacking",
        api_key=OPENAI_API_KEY
    )
    #asyncio.run(website_pipeline(config=ResearchConfig(test_query, test_domain)))

    asyncio.run(website_controller.website_pipeline())    
