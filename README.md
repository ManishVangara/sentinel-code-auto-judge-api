# SentinelCode AutoJudge API

A Secure, AI-Powered Multi-Language Code Execution & Evaluation Service

SentinelCode AutoJudge API is a FastAPI-based, production-ready code execution engine that securely runs and evaluates code across multiple programming languages. It features strong sandboxing, smart language detection, resource monitoring, and optional AI-powered test case generation.

This project is designed for real-world use cases such as online judges, coding platforms, automated graders, and secure code execution services.

---

## Features

### Core Features

Multi-Language Support  
Execute code written in:
- Python
- C
- C++
- Java
- JavaScript (Node.js)
- Go
- Perl  
(Requires respective compilers/interpreters to be installed)

Secure Sandboxing
- Execution time limits
- Memory usage caps
- Restricted system access
- Forbidden pattern detection

Smart Language Detection
- Multi-stage weighted pattern matching
- Syntax validation per language
- Prevents declared vs detected language mismatch

AI-Supported Test Case Generation
- Automatic test case creation using:
  - OpenAI
  - Anthropic (Claude)
  - Hugging Face models  
(API keys required)

Resource Monitoring
- Tracks execution time
- Tracks memory usage using psutil

Flexible Input Handling
- Supports string-based inputs
- Supports array-based test cases
- Suitable for real-world judging scenarios

---

## Security Features and Considerations

- Maximum code size: 10 KB
- Default execution timeout: 5 seconds
- Maximum memory usage: 150 MB
- Detection of forbidden and dangerous operations
- Directory traversal protection
- System file access prevention
- Process isolation and cleanup after execution

---

## Prerequisites

Required:
- Python 3.8 or higher

Language Compilers and Interpreters (as needed):

Language | Requirement
-------- | -----------
C | gcc
C++ | g++
Java | javac, java
JavaScript | node
Go | go
Perl | perl

---

## Python Dependencies

Install required Python libraries:

pip install -r requirements.txt

AI Providers:
- openai
- anthropic
- huggingface_hub

It is strongly recommended to use a Python virtual environment.

---

## Configuration

Create a .env file at the project root to store API keys:

OPENAI_API_KEY=your_openai_key  
ANTHROPIC_API_KEY=your_anthropic_key  
HUGGINGFACE_API_KEY=your_huggingface_key  

You may include one, some, or all providers.

---

## Usage

Start the server:

python main.py

Or using Uvicorn:

uvicorn main:app --host 0.0.0.0 --port 8000

---

## API Endpoints

GET /  
Returns server status and active configuration.

GET /languages  
Returns a list of supported programming languages.

POST /run  
Main endpoint to execute and evaluate code.

Request includes:
- Source code
- Optional declared language
- Optional test cases or input
- Execution preferences

Response includes:
- Execution output
- Errors (if any)
- Runtime
- Memory usage
- Security or validation messages

Note: When sending code as raw JSON strings, ensure proper escaping of special characters.

---

## Common Issues

Issue | Cause | Solution
----- | ----- | --------
Compiler not found | Missing compiler | Install required compiler
Language mismatch | Declared vs detected language differs | Fix language declaration
Security violation | Forbidden pattern detected | Remove unsafe operations
Execution timeout | Code exceeded time limit | Optimize logic
Parsing errors | Improper JSON escaping | Escape characters correctly

---

## Project Flow

1. Request validation
2. Language detection and verification
3. Security checks
4. Test case generation (optional)
5. Secure compilation and execution
6. Resource monitoring
7. Result aggregation and response

![](Code_Compiler_Flow.JPG)
![](Code_Compiler_Flow_2.JPG)

---

## Future Improvements

- Improved language-specific test generation prompts
- More efficient input and test handling
- Enhanced security pattern detection
- Rate limiting for production environments
- Docker-based containerized execution
- Distributed execution support

---

## Project Structure

sentinel-code-auto-judge-api/
|
├── main.py              # Entry point & Monolithic Logic
├── requirements.txt     # Python dependencies
├── README.md            # Documentation
├── Application Report.md # Technical Report
├── standard_tests.json  # Test cases
├── test_cases_all_languages.json # More test cases
└── .gitignore

---

## License

Choose and include an appropriate license (MIT, Apache 2.0, GPL, etc.) before publishing.

---

## Final Note

SentinelCode AutoJudge API is built with security, extensibility, and real-world deployment in mind. It is suitable for educational platforms, automated grading systems, and secure code execution services.
