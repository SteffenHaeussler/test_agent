# Phase 1 Implementation Plan: Security & Reliability

Following TDD and Tidy First principles, this plan separates structural changes from behavioral changes.

## Overview

Phase 1 focuses on critical security vulnerabilities and reliability issues that pose immediate risks to the system. Each task will follow the TDD cycle (Red → Green → Refactor) with clear separation of structural and behavioral changes.

## ✅ COMPLETED: SQL Injection Protection
The SQL injection protection has been fully implemented:
- Created `src/agent/validators/sql_validator.py` with comprehensive validation
- Full test suite in `tests/unit/test_sql_validator.py` (12 tests, all passing)
- Integrated into `src/agent/domain/sql_model.py`
- Uses whitelist approach allowing only SELECT and WITH operations
- Prevents multiple statements, forbidden operations, and UNION attacks

## 1. Error Handling Framework

### Current State Analysis
- **Locations**: Throughout codebase, especially `adapters/`
- **Issue**: Generic Exception catching, silent failures
- **Risk**: Errors are hidden, making debugging difficult

### Implementation Steps (TDD Approach)

#### Step 1: Create Exception Hierarchy (Structural)
1. Create `src/agent/exceptions.py`
2. Define base exception classes (no behavior yet)
3. This is purely structural - organizing code

#### Step 2: Write Tests for Each Exception Type (Red Phase)
```python
# Test 1: DatabaseConnectionError should include connection details
def test_database_connection_error_includes_details()

# Test 2: QueryExecutionError should include query and error
def test_query_execution_error_includes_context()

# Test 3: ValidationError should include field and value
def test_validation_error_includes_field_info()

# Test 4: Exceptions should chain properly
def test_exception_chaining()
```

#### Step 3: Implement Exception Classes (Green Phase)
1. Add properties and __str__ methods to each exception
2. Implement proper exception chaining
3. Add context preservation

#### Step 4: Replace Generic Exceptions (Behavioral)
1. Start with `database.py` - write test for specific exception
2. Replace generic catch with specific exception
3. Run all tests to ensure no regression
4. Repeat for each adapter

#### Step 5: Remove Silent Failures (Behavioral)
1. Find all `return None` on error paths
2. Write test expecting exception
3. Change to raise appropriate exception
4. Verify error propagation

### Commit Plan
- Commit 1: [STRUCTURAL] Create exception hierarchy structure
- Commit 2: [BEHAVIORAL] Implement DatabaseConnectionError
- Commit 3: [BEHAVIORAL] Implement QueryExecutionError
- Commit 4: [BEHAVIORAL] Replace generic exceptions in database.py
- Commit 5: [BEHAVIORAL] Remove silent failures in database.py
- Commit 6: [BEHAVIORAL] Apply same pattern to other adapters

## 2. Database Connection Reliability

### Current State Analysis
- **Location**: `src/agent/adapters/database.py`
- **Issue**: No connection pooling, synchronous operations
- **Risk**: Connection exhaustion, poor performance

### Implementation Steps (TDD Approach)

#### Step 1: Test Current Behavior (Baseline)
1. Write performance test for current implementation
2. Write test for connection limit behavior
3. Document current metrics

#### Step 2: Create Async Database Adapter (Structural)
1. Create `src/agent/adapters/async_database.py`
2. Copy existing interface (no implementation)
3. Create test file mirroring existing tests

#### Step 3: Implement Connection Pooling (Behavioral)
```python
# Test 1: Should limit concurrent connections
def test_connection_pool_limits_connections()

# Test 2: Should recycle old connections
def test_connection_pool_recycles_connections()

# Test 3: Should handle pool exhaustion gracefully
def test_connection_pool_exhaustion_handling()

# Test 4: Should timeout on connection wait
def test_connection_pool_timeout()
```

#### Step 4: Implement Async Operations (Behavioral)
1. Convert one method at a time to async
2. Write async test for each method
3. Ensure backwards compatibility

#### Step 5: Add Retry Logic (Behavioral)
```python
# Test 1: Should retry on connection failure
def test_retry_on_connection_failure()

# Test 2: Should use exponential backoff
def test_exponential_backoff()

# Test 3: Should respect max retries
def test_max_retry_limit()
```

### Commit Plan
- Commit 1: [STRUCTURAL] Add async database adapter structure
- Commit 2: [BEHAVIORAL] Add connection pool configuration
- Commit 3: [BEHAVIORAL] Convert execute_query to async
- Commit 4: [BEHAVIORAL] Add retry logic with exponential backoff
- Commit 5: [STRUCTURAL] Extract retry logic to decorator
- Commit 6: [BEHAVIORAL] Apply retry to all database operations

## 3. Input Validation Framework

### Current State Analysis
- **Issue**: Limited validation on user inputs
- **Risk**: Invalid data can cause crashes or security issues

### Implementation Steps (TDD Approach)

#### Step 1: Create Validation Infrastructure (Structural)
1. Create `src/agent/validators/input_validator.py`
2. Create base validator class
3. Set up test structure

#### Step 2: Implement Question Validation (Behavioral)
```python
# Test 1: Should reject empty questions
def test_reject_empty_questions()

# Test 2: Should reject questions exceeding max length
def test_reject_oversized_questions()

# Test 3: Should sanitize special characters
def test_sanitize_special_characters()

# Test 4: Should validate question structure
def test_validate_question_structure()
```

#### Step 3: Add Rate Limiting (Behavioral)
1. Implement token bucket algorithm
2. Test rate limit enforcement
3. Test rate limit recovery

### Commit Plan
- Commit 1: [STRUCTURAL] Create validation infrastructure
- Commit 2: [BEHAVIORAL] Add question validation
- Commit 3: [BEHAVIORAL] Add rate limiting
- Commit 4: [STRUCTURAL] Extract rate limiter to middleware

## Execution Timeline

### Week 1
- Day 1-2: Error Handling Framework
- Day 3-4: Database Connection Reliability (Part 1)
- Day 5: Integration testing and fixes

### Week 2
- Day 1-2: Database Connection Reliability (Part 2)
- Day 3-4: Input Validation Framework
- Day 5: Final integration and testing

## Success Criteria

Each implementation must meet:
1. All tests passing (100% of new tests)
2. No regression in existing tests
3. Test coverage ≥ 90% for new code
4. All linter warnings resolved
5. Clear separation of structural/behavioral commits
6. Performance benchmarks show no degradation

## Risk Mitigation

1. **Feature Flags**: Add feature flags for new validation
2. **Rollback Plan**: Each commit should be independently revertible
3. **Monitoring**: Add metrics for validation rejections
4. **Gradual Rollout**: Deploy to staging first, monitor for issues

## Dependencies

- `asyncpg` for async PostgreSQL operations
- `pytest-asyncio` for async testing
- Existing test infrastructure must be working

## Notes

- Run `make test` after each commit
- Use `make coverage` to verify test coverage
- Follow existing code style and patterns
- Document any deviations from the plan
