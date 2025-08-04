# TODO: Codebase Improvements

## 🚨 Phase 1: Critical Security & Reliability (Week 1-2)

### Security Hardening
- [ ] **SQL Injection Protection** - `src/agent/domain/sql_model.py`
  - [ ] Create SQLValidator class with whitelist of allowed operations (SELECT, WITH only)
  - [ ] Add SQL parsing and validation before execution
  - [ ] Implement query parameter sanitization
  - [ ] Add audit logging for all SQL executions

- [ ] **Input Validation Framework**
  - [ ] Add Pydantic validators for all user inputs
  - [ ] Implement rate limiting on API endpoints
  - [ ] Add request size limits
  - [ ] Validate file uploads and external data sources

### Database Reliability
- [ ] **Connection Pool Configuration** - `src/agent/adapters/database.py`
  - [ ] Configure SQLAlchemy connection pooling (pool_size=20, max_overflow=30)
  - [ ] Add connection timeout and recycling
  - [ ] Implement connection health checks
  - [ ] Add retry logic with exponential backoff

- [ ] **Async Database Operations**
  - [ ] Replace `create_engine` with `create_async_engine`
  - [ ] Convert all database operations to async
  - [ ] Add proper connection context managers
  - [ ] Implement query timeout handling

### Error Handling
- [ ] **Custom Exception Hierarchy** - Create `src/agent/exceptions.py`
  ```python
  DatabaseConnectionError
  QueryExecutionError
  ValidationError
  AgentProcessingError
  ToolExecutionError
  ```

- [ ] **Fix Error Handling Patterns**
  - [ ] Replace generic `except Exception` with specific exceptions
  - [ ] Remove silent failures (returning None)
  - [ ] Add proper error propagation through message bus
  - [ ] Implement structured error logging

## 🏗️ Phase 2: Architecture Refactoring (Week 3-4)

### State Machine Improvements
- [ ] **Refactor BaseAgent.update()** - `src/agent/domain/model.py`
  - [ ] Extract command handlers to strategy pattern
  - [ ] Create CommandHandlerRegistry
  - [ ] Remove large switch statement
  - [ ] Add handler registration mechanism

- [ ] **Decompose Large Methods**
  - [ ] Break down `create_prompt()` (66 lines) into smaller functions
  - [ ] Extract prompt building logic
  - [ ] Separate context preparation from prompt formatting

### Code Duplication Removal
- [ ] **Extract Shared Utilities** - Create `src/agent/utils/`
  - [ ] Move `convert_schema()` to shared utility
  - [ ] Create base handler class for common handler logic
  - [ ] Extract common adapter patterns
  - [ ] Consolidate logging configuration

- [ ] **Configuration Management** - Create `src/agent/config_manager.py`
  - [ ] Centralize environment variable handling
  - [ ] Add configuration validation on startup
  - [ ] Create environment-specific config classes
  - [ ] Replace magic strings with constants

### Design Pattern Improvements
- [ ] **Implement Proper DI Container**
  - [ ] Replace manual dependency injection in bootstrap.py
  - [ ] Add lifecycle management for resources
  - [ ] Implement proper factory patterns
  - [ ] Add dependency validation

## ⚡ Phase 3: Performance Optimization (Week 5-6)

### Async/Await Consistency
- [ ] **Full Async Implementation**
  - [ ] Convert all I/O operations to async
  - [ ] Add async LLM client implementations
  - [ ] Implement async message bus
  - [ ] Use asyncio for concurrent operations

- [ ] **Memory Optimization**
  - [ ] Add pagination limits to prevent unbounded growth
  - [ ] Implement streaming for large datasets
  - [ ] Add result set size limits
  - [ ] Implement proper garbage collection hints

### Caching Strategy
- [ ] **Implement Caching Layer**
  - [ ] Add Redis for distributed caching
  - [ ] Cache LLM responses where appropriate
  - [ ] Implement cache invalidation strategy
  - [ ] Add cache metrics and monitoring

## 📊 Phase 4: Testing & Monitoring (Week 7-8)

### Test Coverage
- [ ] **Expand Test Suite**
  - [ ] Add negative test cases for all error paths
  - [ ] Implement concurrent scenario testing
  - [ ] Add integration tests for async operations
  - [ ] Create performance regression tests

- [ ] **Testing Infrastructure**
  - [ ] Add test fixtures for database operations
  - [ ] Implement proper test data factories
  - [ ] Add property-based testing for validators
  - [ ] Create end-to-end test scenarios

### Observability
- [ ] **Enhanced Monitoring**
  - [ ] Add structured logging with correlation IDs
  - [ ] Implement distributed tracing properly
  - [ ] Add performance metrics collection
  - [ ] Create alerting rules for critical paths

## 🔧 Phase 5: Code Quality (Ongoing)

### Type Safety
- [ ] **Complete Type Annotations**
  - [ ] Add missing type hints throughout codebase
  - [ ] Use proper generic types (Dict[str, Any] not Dict)
  - [ ] Add mypy to CI pipeline
  - [ ] Implement runtime type checking for critical paths

### Documentation
- [ ] **Technical Documentation**
  - [ ] Add docstrings to all public methods
  - [ ] Create architecture decision records (ADRs)
  - [ ] Document error handling strategies
  - [ ] Add API documentation with examples

### Dependency Management
- [ ] **Optimize Dependencies**
  - [ ] Audit and remove unused dependencies
  - [ ] Evaluate lighter alternatives for heavy libraries
  - [ ] Pin all dependency versions
  - [ ] Add security scanning for dependencies

## 📋 Quick Wins (Can be done anytime)

- [ ] Replace magic strings with named constants
- [ ] Add `.pre-commit-config.yaml` for code quality
- [ ] Configure proper logging levels
- [ ] Add health check endpoints
- [ ] Implement graceful shutdown handling
- [ ] Add request ID tracking
- [ ] Create development setup script
- [ ] Add performance profiling decorators

## 🎯 Success Metrics

- [ ] Zero SQL injection vulnerabilities
- [ ] 90%+ test coverage
- [ ] <100ms p95 response time for simple queries
- [ ] Zero unhandled exceptions in production
- [ ] All critical paths have proper error handling
- [ ] Full async implementation with no blocking I/O

## 📝 Notes

- Prioritize security fixes first (SQL injection, input validation)
- Each phase builds on the previous one
- Consider feature freeze during Phase 1 & 2
- Implement monitoring before major refactoring
- Keep backwards compatibility where possible