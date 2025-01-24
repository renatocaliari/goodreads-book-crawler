import asyncio
import json
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, LLMExtractionStrategy
from composio_phidata import Action, App, ComposioToolSet

import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.utils.log import logger
import asyncio
from typing import List, Dict
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import re
from pydantic import BaseModel, Field, AnyUrl
from decimal import Decimal
from phi.agent import Agent, RunResponse
import os
import pandas as pd
# from rich.console import Console
# from rich.table import Table
import logging

# os.system('playwright install')
# os.system('playwright install-deps')
# os.system('crawl4ai-setup')

# Suppress logging warnings
# os.environ["GRPC_VERBOSITY"] = "ERROR"
# os.environ["GLOG_minloglevel"] = "2"

browser_config = BrowserConfig(
    headless=True,
    # For better performance in Docker or low-memory environments:
    extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
)

crawl_config = CrawlerRunConfig(
    markdown_generator=DefaultMarkdownGenerator(),
    exclude_external_links=True,    # Remove external links
    remove_overlay_elements=True,   # Remove popups/modals
    process_iframes=False           # Process iframe conten
)
session_id = "session1"  # Reuse the same session across all URLs
crawler = AsyncWebCrawler(config=browser_config)

class LoggerManager:
    def __init__(self, logger, main_status_placeholder=None):
        self.logger = logger
        self.main_status_placeholder = main_status_placeholder

    def log(self, message):
        self.logger.info(message)
        if self.main_status_placeholder:
            self.main_status_placeholder.write(message)

class Book(BaseModel):
    title: str = Field(..., description="Title of the book.")
    book_url: AnyUrl = Field(..., description="The link url of the book.")
    summary: str = Field(..., description="Summary of the book.")
    author: str = Field(..., description="Author of the book.") 
    pages: int = Field(..., description="Number of pages of the book.")
    rating: Decimal = Field(..., description="Rating of the book with two decimal places.", ge=0.01, decimal_places=2)
    # float = Field(..., description="Rating of the book with two decimal places.")
    genres: list[str] = Field(..., description="List of genres.")
    reviews: int  = Field(..., description="Number of reviews of the book.")

class ListBooks(BaseModel):
    books: list[Book] = Field(..., description="List of books.")

# def display_books_table(books: List[Book]):
#     """Displays a list of books in a formatted table using rich."""
#     console = Console()
#     table = Table(title="Books Information")

#     table.add_column("Title", style="cyan", no_wrap=True)
#     table.add_column("Author", style="magenta")
#     table.add_column("Summary", style="green", no_wrap=True)
#     table.add_column("Pages", style="yellow", justify="right")
#     table.add_column("Rating", style="blue", justify="right")
#     table.add_column("Reviews", style="red", justify="right")
    
#     for book in books:
#         table.add_row(
#             book.title if book.title else "Unknown",
#             book.author if book.author else "Unknown",
#             book.summary if book.summary else "No summary",
#             ", ".join(book.genres) if book.genres else "-",
#             str(book.pages) if book.pages else "Unknown",
#             str(book.rating) if book.rating else "Unknown",
#             str(book.reviews) if book.reviews else "Unknown"
#         )
#     console.print(table)

def save_books_to_csv(df_books: pd.DataFrame, filename: str = "books.csv"):
    """Saves a list of books to a CSV file."""
    df_books.to_csv(filename, index=False)
    print(f"Books saved to '{filename}'")


def get_info_from_book(book_data: str, progress_text: str):
    book_agent: Agent = Agent(model=MODEL_GEMINI,
            description=f"Extract the book information.",
            response_model=Book,
            session_id=session_id)

    logger_manager.log(f"🚀 Extracting data from book | {progress_text}")
    response: RunResponse = book_agent.run(book_data)
    # logger_manager.log(f"🚀 Response from agent: {response}")
    return response.content

async def crawl_sequential(urls: List[str], max_urls=None) -> ListBooks:
    logger_manager.log("\n=== Sequential Crawling with Session Reuse ===")

    if max_urls is None:
        max_urls = len(urls)

    try:
        schema = {
            "name": "Main Content",
            "baseSelector": ".BookPage__rightColumn",    # Repeated elements
            "fields": [
                {
                    "name": "main",
                    "selector": ".BookPage__mainContent",
                    "type": "text"
                }
            ]
        }

        extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

        crawl_config = CrawlerRunConfig(
            # e.g., pass js_code or wait_for if the page is dynamic
            # wait_for="css:.crypto-row:nth-child(20)"
            # cache_mode = CacheMode.BYPASS,
            extraction_strategy=extraction_strategy,
            markdown_generator=DefaultMarkdownGenerator(),
            exclude_external_links=True,
            remove_overlay_elements=True,
            process_iframes=False,
            semaphore_count=5
        )

        # logger_manager.log(f"Crawling urls...")
        list_books: ListBooks = ListBooks(books=[])
        session_id = "my-session"
        results = await crawler.arun_many(
            urls=list(urls)[:max_urls],
            config=crawl_config,
            session_id=session_id,
            verbose=True
        )
        # logger_manager.log(f"results: {results}")

        for index, r in enumerate(results):
        # for index, url in enumerate(list(urls)[:max_urls]):
            progress = f"{index + 1}/{max_urls}"
            logger_manager.log(f"Crawling url: {r.url} | {progress}")
            # session_id = "my-session"
            # result = await crawler.arun(
            #     url=url,
            #     config=crawl_config,
            #     session_id=session_id
            # )
            # if result.success:
                # print(f"Successfully crawled: {url}")
            logger_manager.log(f"Getting structured data from book: {r.url} - {progress}")
            book: Book = get_info_from_book(r.markdown, progress)
            print(f"Book: {book}")
            # logger_manager.log(f"Successfully got details from book: {result.extracted_content}")
            logger_manager.log(f"Successfully got details from book: {book.title}")
            list_books.books.append(book)
            # else:
            #     logger_manager.log(f"Failed to get details from book: {book.title}")
            #     print(f"Failed: {url} - Error: {result.error_message}")
        logger_manager.log(f"Successfully got details from books")
    except Exception as e:
        logger_manager.log(f"Error: {e}")

    finally:
        print("Closing crawler")
        # After all URLs are done, close the crawler (and the browser)
        return list_books

async def get_books_urls(url_books: List[str]):
    try:
        logger_manager.log('Starting getting books urls')
        result = await crawler.arun(
            url=url_books,
            config=crawl_config
            )
        if result.success:
            logger_manager.log('Successfully got books urls')
            print(f"Successfully crawled")
            # E.g. check markdown length
            # print(f"Markdown length: {len(result.markdown_v2.raw_markdown)}")
        else:
            logger_manager.log('Failed getting books urls')
            print(f"Failed - Error: {result.error_message}")
    finally:
        # After all URLs are done, close the crawler (and the browser)
        return result.links
        

def filter_urls(links: List[Dict[str, str]], regex: str) -> List[str]:
    """Filters a list of links based on a regular expression."""
    filtered_urls = []
    for link in links:
        if 'href' in link and re.search(regex, link['href']):
            filtered_urls.append(link['href'])
    return filtered_urls

async def scrape_books(url_books: List[str], max_books: int = 1) -> ListBooks:
    books: ListBooks = ListBooks(books=[])
    try:
        await crawler.start()
        urls = await get_books_urls(url_books)
        regex = r"https://www.goodreads.com/book/show/.*"
        filtered_urls = filter_urls(urls["internal"], regex)
        logger_manager.log(f"url books: {filtered_urls}")
        books = await crawl_sequential(filtered_urls,max_books)
        # logger_manager.log(f"books: {books}")
    except Exception as e:
        logger_manager.log(f"Error {e}")
    finally:
        await crawler.close()
        return books


st.title("Goodreads Books Scraper")

# Sidebar for inputs
with st.sidebar:
    user_api_key = None
    api_key_option = st.selectbox("🤖 AI LLM Gemini API Key:", ("Use Default Key", "Enter Custom Key"))
    if api_key_option == "Enter Custom Key":
        user_api_key = st.text_input("Enter your Gemini API Key:", type="password")
        if not user_api_key:
            st.error("You need to enter a Gemini API key.")
            st.stop()
    elif api_key_option == "Use Default Key":
        if not os.getenv("GEMINI_API_KEY"):
            st.error("The Gemini API key is not set. Enter a custom key or configure the .env file.")
            st.stop()

    api_key = user_api_key or os.getenv("GEMINI_API_KEY")

    MODEL_GEMINI: Gemini = Gemini(id="gemini-2.0-flash-exp", api_key=api_key)

    url_books = st.text_input(
        "Goodreads URL", key="url_books",
        placeholder="Paste the URL here",
        value="https://www.goodreads.com/shelf/show/2025-challenge",
        help="Paste the link of a URL from Goodreads containing a list of books.",
    )

    qty_books = st.number_input("Max books to crawl (1 to 50)", min_value=1, max_value=50, value=10, step=1, label_visibility="visible")
    
    is_button_disabled = not url_books or not qty_books

    if st.button("Get List Details", disabled=is_button_disabled, use_container_width=True):
        st.session_state['run_evaluation'] = True
    else:
        st.session_state['run_evaluation'] = False

main_status_placeholder = st.empty()
logger_manager = LoggerManager(logger, main_status_placeholder)

if st.session_state.get('run_evaluation', False):
    with st.spinner('Scraping and crawling books...'):
        books: ListBooks = asyncio.run(scrape_books(url_books, qty_books))
        st.session_state['books'] = books
        if books and books.books and len(books.books) > 0:
            df = pd.DataFrame([book.model_dump() for book in books.books])
            df.round(2)
            df=df.round({"rating": 2})
            save_books_to_csv(df)
            # display_books_table(books.books)
            st.dataframe(df,
                column_config={
                    "book_url": st.column_config.LinkColumn("Book URL")
                }
            )

            # toolset = ComposioToolSet(entity_id="myself", logging_level=logging.DEBUG)
            # tools = toolset.get_tools(actions=['GOOGLESHEETS_SHEET_FROM_JSON'])
            # agent = Agent(model=MODEL_GEMINI,
            #     description=f"You are a helpful assistant.",
            #     instructions=["Add the list of the books them to a google sheet title 'My books {timestamp}'."],
            #     tools=tools, show_tool_calls=True, verbose=True)
            # books = [
            #     {k: str(v) for k, v in book.model_dump().items()}
            #     for book in books.books
            # ]
            # # Convertendo a lista de livros para JSON
            # books_json = json.dumps(books)
            # logger_manager.log(f"books: {books_json}")
            
            # response: RunResponse = agent.run(books_json)
            # logger_manager.log("rodou agent...")
            # print("🚀 Response from agent: ", response)
        else:
            st.markdown("Error: No books found.")  
            print("No books found to display or save.")

