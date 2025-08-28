from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
import os

# Global index to avoid recreating
_qa_index = None
PERSIST_DIR = "./qa_index"  # directory where index will be stored

def reset_qa_index():
    """Reset the QA index cache to force recreation with new API key."""
    global _qa_index
    _qa_index = None
    print("RAG index cache cleared. Next request will create new index with current API key.")

def get_qa_index():
    """Get or create QA knowledge base (singleton pattern)."""
    global _qa_index
    if _qa_index is None:
        # Force a one-time recreation of the index to fix configuration issues.
        print("Forcing fresh index creation to resolve model configuration errors...")
        _qa_index = create_qa_knowledge_base()
    return _qa_index

def create_qa_knowledge_base():
    """Create and persist the knowledge base for QA testing."""

    # Define your documents here (omitted for brevity)
    documents = [
        # Basic Test Case Structure - Document explaining test case format and structure
        Document(text="Test case structure: ID (TC_MODULE_001), title, preconditions, test steps, expected results, priority (P0/P1/P2), test data. Always use specific, actionable steps starting with verbs like 'Click', 'Enter', 'Verify', 'Navigate', 'Select'."),
        
        # API Testing Templates - Document covering API testing methodologies
        Document(text="API test cases: 1) Functional - verify correct response for valid input, status codes (200, 201, 400, 404, 500), response format (JSON/XML), required fields presence. 2) Authentication - valid/invalid API keys, token expiration, OAuth flows. 3) Input validation - boundary values, special characters, SQL injection, XSS attempts. 4) Error handling - malformed requests, missing parameters, invalid data types."),
        
        # API Performance Testing - Document covering performance testing for APIs
        Document(text="API performance testing: Response time <2 seconds, concurrent user handling (100+ users), rate limiting validation, timeout scenarios, payload size limits. Monitor CPU, memory, database connections during load tests."),
        
        # API Security Patterns - Document covering security testing for APIs
        Document(text="API security patterns: Authentication bypass attempts, authorization checks (RBAC), input sanitization, SQL injection ('; DROP TABLE), XSS prevention, CORS validation, HTTPS enforcement, API versioning security."),
        
        # UI Testing Templates - Document covering user interface testing
        Document(text="UI functional test cases: Element visibility, click functionality, form validation, navigation flow, responsive design across devices, browser compatibility (Chrome, Firefox, Safari, Edge), accessibility compliance (WCAG 2.1)."),
        
        # UI Validation Patterns - Document covering UI input validation
        Document(text="UI validation patterns: Required field validation, data format validation (email, phone, date), character limits, special character handling, file upload restrictions, dropdown selections, checkbox/radio button states."),
        
        # UI Performance Criteria - Document covering UI performance standards
        Document(text="UI performance criteria: Page load time <3 seconds, image optimization, lazy loading, CSS/JS minification, browser caching, mobile responsiveness, touch gestures, orientation changes."),
        
        # Integration Testing Templates - Document covering system integration testing
        Document(text="Integration test patterns: API-to-API communication, database connectivity, third-party service integration, message queue processing, file system operations, email/SMS service integration, payment gateway integration."),
        
        # Integration Test Scenarios - Document covering specific integration test cases
        Document(text="Integration test scenarios: Data flow validation, error propagation, transaction rollback, retry mechanisms, circuit breaker patterns, service timeout handling, dependency failure scenarios."),
        
        # Security Testing Comprehensive Checklist - Document covering security testing
        Document(text="Security testing checklist: 1) Authentication - multi-factor authentication, password complexity, account lockout, session timeout. 2) Authorization - role-based access control, privilege escalation, horizontal/vertical access control. 3) Input validation - SQL injection, XSS, LDAP injection, command injection, file upload validation."),
        
        # Security Vulnerability Patterns - Document covering OWASP Top 10 vulnerabilities
        Document(text="Security vulnerability patterns: OWASP Top 10 - injection attacks, broken authentication, sensitive data exposure, XML external entities (XXE), broken access control, security misconfiguration, cross-site scripting, insecure deserialization, known vulnerabilities, insufficient logging."),
        
        # Security Test Cases - Document covering specific security test scenarios
        Document(text="Security test cases: Password strength validation, SQL injection attempts ('OR 1=1--), XSS payload injection (<script>alert('XSS')</script>), file upload malware, session hijacking, CSRF token validation, HTTPS enforcement, data encryption at rest and in transit."),
        
        # Performance Testing Criteria - Document covering performance testing types
        Document(text="Performance testing types: Load testing (normal expected load), stress testing (beyond normal capacity), spike testing (sudden load increases), volume testing (large amounts of data), endurance testing (extended periods)."),
        
        # Performance Metrics and Thresholds - Document covering performance standards
        Document(text="Performance metrics and thresholds: Response time <2 seconds (web), <1 second (mobile), throughput >1000 requests/second, CPU utilization <80%, memory usage <85%, error rate <1%, 99.9% uptime requirement, concurrent user support 100+ users."),
        
        # Performance Test Scenarios - Document covering specific performance test cases
        Document(text="Performance test scenarios: Peak traffic simulation, database connection pool exhaustion, memory leak detection, cache performance, CDN effectiveness, auto-scaling validation, graceful degradation under load."),
        
        # Edge Cases and Negative Testing - Document covering edge case testing
        Document(text="Edge case testing patterns: Boundary value testing (min/max values), null/empty input handling, special character processing, unicode support, date/time edge cases (leap year, timezone), numeric overflow/underflow, concurrent operations, race conditions."),
        
        # Negative Test Scenarios - Document covering negative testing approaches
        Document(text="Negative test scenarios: Invalid authentication, insufficient permissions, malformed data input, network connectivity issues, service unavailability, database connection failures, file system errors, third-party service timeouts."),
        
        # Data Corruption Scenarios - Document covering data integrity testing
        Document(text="Data corruption scenarios: Incomplete transactions, power failure simulation, network interruption during data transfer, concurrent data modification, orphaned records, referential integrity violations, backup/restore validation."),
        
        # Accessibility Testing - Document covering accessibility testing criteria
        Document(text="Accessibility testing criteria: Keyboard navigation support, screen reader compatibility, color contrast ratios (4.5:1 normal text, 3:1 large text), alt text for images, ARIA labels, focus indicators, text resizing up to 200%, semantic HTML structure."),
        
        # WCAG 2.1 Compliance Patterns - Document covering accessibility standards
        Document(text="WCAG 2.1 compliance patterns: Perceivable (text alternatives, captions), Operable (keyboard accessible, no seizures), Understandable (readable, predictable), Robust (compatible with assistive technologies)."),
        
        # Mobile Testing Patterns - Document covering mobile app testing
        Document(text="Mobile testing scenarios: Touch gestures (tap, swipe, pinch), device orientation changes, network switching (WiFi/cellular), background/foreground transitions, push notifications, battery optimization, storage limitations, camera/GPS integration."),
        
        # Mobile Performance Criteria - Document covering mobile performance standards
        Document(text="Mobile performance criteria: App launch time <3 seconds, smooth scrolling (60 FPS), battery consumption optimization, data usage minimization, offline functionality, sync capabilities, crash-free sessions >99.5%."),
        
        # Database Testing Templates - Document covering database testing
        Document(text="Database test patterns: CRUD operations validation, data integrity checks, foreign key constraints, transaction isolation levels, deadlock detection, backup/restore procedures, data migration validation, query performance optimization."),
        
        # Database Performance Testing - Document covering database performance
        Document(text="Database performance testing: Query execution time, connection pool management, index effectiveness, concurrent user impact on database, data archiving procedures, replication lag monitoring."),
        
        # Test Case Categories by Requirement Type - Document covering requirement-based testing
        Document(text="Functional requirement test cases: User story acceptance criteria, business rule validation, workflow testing, feature functionality, input/output validation, calculation accuracy, report generation."),
        
        # Non-functional Requirement Test Cases - Document covering non-functional testing
        Document(text="Non-functional requirement test cases: Performance benchmarks, scalability limits, security compliance, usability standards, reliability metrics, maintainability aspects, portability requirements."),
        
        # Error Handling Patterns - Document covering error handling testing
        Document(text="Error handling test patterns: Graceful error messages, error logging, user-friendly error displays, error recovery mechanisms, retry logic, fallback procedures, error notification systems, audit trails."),
        
        # Compliance and Regulatory Testing - Document covering compliance testing
        Document(text="Compliance testing patterns: GDPR data protection, PCI-DSS payment security, HIPAA healthcare privacy, SOX financial controls, accessibility standards (WCAG, Section 508), industry-specific regulations."),
    ]

    # Create vector index from documents with latest llama-index syntax
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        show_progress=True,
    )

    # Persist the index so we don't re-embed every run
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"QA index created and saved at {PERSIST_DIR}")

    return index
