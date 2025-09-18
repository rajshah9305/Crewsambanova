import os
import threading
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from dotenv import load_dotenv
import uuid
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("SESSION_SECRET", "your-secret-key-here")

# Server-side task store (thread-safe dictionary)
task_store = {}
task_store_lock = threading.Lock()

# Configure SambaNova client with OpenAI compatibility
client = OpenAI(
    api_key=os.getenv("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

def create_sambanova_llm():
    """Create a configured SambaNova LLM for CrewAI agents"""
    return LLM(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
        temperature=0.1,
        top_p=0.1
    )

def validate_api_key():
    """Validate that the SambaNova API key is available"""
    api_key = os.getenv("SAMBANOVA_API_KEY")
    if not api_key:
        logger.error("SAMBANOVA_API_KEY not found in environment variables")
        raise ValueError("SAMBANOVA_API_KEY is required but not found in environment")

# Create shared LLM instance
sambanova_llm = create_sambanova_llm()

# Define Agent Roles
researcher = Agent(
    role='Research Analyst',
    goal='Gather comprehensive information and analyze data on given topics',
    backstory="""You are an expert research analyst with years of experience in 
    data collection, analysis, and synthesis. You excel at finding relevant 
    information from various sources and presenting clear, actionable insights.""",
    verbose=True,
    allow_delegation=False,
    llm=sambanova_llm
)

writer = Agent(
    role='Content Writer',
    goal='Create engaging, well-structured content based on research findings',
    backstory="""You are a skilled content writer who specializes in transforming 
    complex research data into clear, engaging, and accessible content. You have 
    a talent for storytelling and making technical information understandable.""",
    verbose=True,
    allow_delegation=False,
    llm=sambanova_llm
)

reviewer = Agent(
    role='Quality Reviewer',
    goal='Review and enhance content for accuracy, clarity, and engagement',
    backstory="""You are a meticulous quality reviewer with an eye for detail. 
    You ensure that all content meets high standards of accuracy, clarity, and 
    engagement while maintaining consistency in tone and style.""",
    verbose=True,
    allow_delegation=False,
    llm=sambanova_llm
)

def create_research_task(topic):
    """Create a research task for the given topic"""
    return Task(
        description=f"""Research the topic: {topic}
        
        Your task is to:
        1. Gather comprehensive information about {topic}
        2. Analyze key trends, developments, and insights
        3. Identify the most important points and findings
        4. Organize the information in a structured format
        
        Provide a detailed research report with key findings and insights.""",
        agent=researcher,
        expected_output="A comprehensive research report with key findings and insights"
    )

def create_writing_task():
    """Create a writing task based on research findings"""
    return Task(
        description="""Based on the research findings, create engaging content.
        
        Your task is to:
        1. Transform the research data into compelling content
        2. Structure the information logically
        3. Use clear, accessible language
        4. Include relevant examples and insights
        5. Ensure the content is engaging and informative
        
        Create a well-structured article or report.""",
        agent=writer,
        expected_output="A well-written, engaging article based on the research findings"
    )

def create_review_task():
    """Create a review task to enhance the written content"""
    return Task(
        description="""Review and enhance the written content.
        
        Your task is to:
        1. Check for accuracy and consistency
        2. Improve clarity and readability
        3. Enhance engagement and flow
        4. Ensure proper structure and organization
        5. Provide final polished version
        
        Deliver the final, publication-ready content.""",
        agent=reviewer,
        expected_output="Final, polished, and publication-ready content"
    )

def run_crew_workflow(topic, task_id):
    """Run the complete CrewAI workflow with SambaNova models"""
    try:
        logger.info(f"Starting CrewAI workflow for task {task_id} with topic: {topic}")
        
        # Update status to processing
        with task_store_lock:
            task_store[task_id] = {'status': 'processing', 'topic': topic}
        
        # Create tasks
        research_task = create_research_task(topic)
        writing_task = create_writing_task()
        review_task = create_review_task()
        
        # Create crew
        crew = Crew(
            agents=[researcher, writer, reviewer],
            tasks=[research_task, writing_task, review_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute the workflow
        result = crew.kickoff()
        
        # Store result in task store
        with task_store_lock:
            task_store[task_id] = {
                'status': 'completed',
                'result': str(result),
                'topic': topic
            }
        
        logger.info(f"CrewAI workflow completed successfully for task {task_id}")
        return str(result)
        
    except Exception as e:
        logger.error(f"CrewAI workflow failed for task {task_id}: {str(e)}")
        with task_store_lock:
            task_store[task_id] = {
                'status': 'error',
                'error': str(e),
                'topic': topic
            }
        return None

@app.route('/')
def index():
    """Main page with form to submit research topics"""
    return render_template('index.html')

@app.route('/start_research', methods=['POST'])
def start_research():
    """Start the CrewAI research workflow"""
    topic = request.form.get('topic')
    if not topic:
        return jsonify({'error': 'Topic is required'}), 400
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Initialize status in task store
    with task_store_lock:
        task_store[task_id] = {'status': 'initializing', 'topic': topic}
    
    # Start the workflow in a background thread
    thread = threading.Thread(target=run_crew_workflow, args=(topic, task_id))
    thread.daemon = True
    thread.start()
    
    logger.info(f"Started research task {task_id} for topic: {topic}")
    return jsonify({'task_id': task_id, 'status': 'initializing'})

@app.route('/check_status/<task_id>')
def check_status(task_id):
    """Check the status of a research task"""
    with task_store_lock:
        task_data = task_store.get(task_id)
    
    if not task_data:
        return jsonify({'status': 'not_found', 'error': 'Task not found'}), 404
    
    status = task_data.get('status', 'unknown')
    
    if status == 'completed':
        result = task_data.get('result', '')
        return jsonify({'status': status, 'result': result})
    elif status == 'error':
        error = task_data.get('error', 'Unknown error')
        return jsonify({'status': status, 'error': error})
    else:
        return jsonify({'status': status})

@app.route('/test_api')
def test_api():
    """Test SambaNova API connection"""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello! Please confirm that the SambaNova API is working correctly."}
            ],
            temperature=0.1,
            top_p=0.1
        )
        
        return jsonify({
            'status': 'success',
            'response': response.choices[0].message.content
        })
        
    except Exception as e:
        logger.error(f"SambaNova API test failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Validate API key at startup
    try:
        validate_api_key()
        logger.info("SambaNova API key validation successful")
    except ValueError as e:
        logger.error(f"Startup failed: {str(e)}")
        exit(1)
    
    logger.info("Starting Flask web application...")
    app.run(host='0.0.0.0', port=5000, debug=True)