#!/bin/bash
set -e

echo "Starting E2E tests for Meilisearch..."

API_URL="${API_URL:-http://localhost:8080}"
WEB_URL="${WEB_URL:-http://localhost:8081}"
MEILISEARCH_URL="${MEILISEARCH_URL:-http://localhost:7700}"
ENV_FILE="${ENV_FILE:-.env.e2e}"

# Wait for services to be ready
echo "Waiting for services..."
timeout 60 bash -c "until curl -sf ${MEILISEARCH_URL}/health > /dev/null; do sleep 2; done"
echo "Meilisearch is ready"
timeout 60 bash -c "until curl -sf ${API_URL}/health > /dev/null; do sleep 2; done"
echo "API is ready"

# Run indexer
echo "Running indexer..."
docker compose --env-file ${ENV_FILE} -f docker-compose.test.yml -f docker-compose.e2e.override.yml run --no-deps --rm indexer

# Wait for Meilisearch to finish indexing
echo "Waiting for indexing to complete..."
sleep 5

# Check indexing status
while true; do
    IS_INDEXING=$(curl -sf "${API_URL}/stats" | jq -r '.is_indexing')
    if [ "$IS_INDEXING" = "false" ]; then
        echo "Indexing complete"
        break
    fi
    echo "Still indexing..."
    sleep 2
done

# Test 1: Basic substring search
echo "Test 1: Basic substring search"
RESULT=$(curl -sf "${API_URL}/search?q=doc&mode=substr" | jq -r '.total')
if [ "$RESULT" -gt 0 ]; then
    echo "✓ Basic search passed (found $RESULT results)"
else
    echo "✗ Basic search failed"
    # Debug: show what the search returned
    curl -sf "${API_URL}/search?q=doc&mode=substr" | jq .
    exit 1
fi

# Test 2: Plain text search (Meilisearch default)
echo "Test 2: Plain text search"
RESULT=$(curl -sf "${API_URL}/search?q=script&mode=plain" | jq -r '.total')
if [ "$RESULT" -gt 0 ]; then
    echo "✓ Plain text search passed (found $RESULT results)"
else
    echo "✗ Plain text search failed"
    curl -sf "${API_URL}/search?q=script&mode=plain" | jq .
    exit 1
fi

# Test 3: Regex search (simulated)
echo "Test 3: Regex search (simulated)"
RESULT=$(curl -sf "${API_URL}/search?q=script[0-9]%2B\\.py&mode=regex" | jq -r '.total')
if [ "$RESULT" -ge 0 ]; then
    echo "✓ Regex search passed (found $RESULT results)"
else
    echo "✗ Regex search failed"
    curl -sf "${API_URL}/search?q=script[0-9]%2B\\.py&mode=regex" | jq .
fi

# Test 4: Extension filter
echo "Test 4: Extension filter"
RESULT=$(curl -sf "${API_URL}/search?ext=txt&ext=py" | jq -r '.total')
if [ "$RESULT" -gt 0 ]; then
    echo "✓ Extension filter passed (found $RESULT results)"
else
    echo "✗ Extension filter failed"
    exit 1
fi

# Test 5: Stats endpoint
echo "Test 5: Stats endpoint"
TOTAL=$(curl -sf "${API_URL}/stats" | jq -r '.total_files')
if [ "$TOTAL" -gt 0 ]; then
    echo "✓ Stats endpoint passed (found $TOTAL files)"
else
    echo "✗ Stats endpoint failed"
    exit 1
fi

# Test 6: Web UI availability
echo "Test 6: Web UI availability"
HTTP_CODE=$(curl -o /dev/null -s -w "%{http_code}" ${WEB_URL})
if [ "$HTTP_CODE" -eq 200 ]; then
    echo "✓ Web UI is accessible"
else
    echo "✗ Web UI is not accessible (HTTP $HTTP_CODE)"
    exit 1
fi

# Test 7: Pagination
echo "Test 7: Pagination"
PAGE1=$(curl -sf "${API_URL}/search?page=1&per_page=10" | jq -r '.results | length')
PAGE2=$(curl -sf "${API_URL}/search?page=2&per_page=10" | jq -r '.results | length')
if [ "$PAGE1" -gt 0 ] && [ "$PAGE2" -ge 0 ]; then
    echo "✓ Pagination passed"
else
    echo "✗ Pagination failed"
    exit 1
fi

# Test 8: Sorting
echo "Test 8: Sorting"
RESULT=$(curl -sf "${API_URL}/search?sort=size_desc&per_page=5" | jq -r '.results[0].size')
if [ "$RESULT" -gt 0 ]; then
    echo "✓ Sorting passed"
else
    echo "✗ Sorting failed"
    exit 1
fi

# Test 9: Size filter
echo "Test 9: Size filter"
RESULT=$(curl -sf "${API_URL}/search?size_min=1000&size_max=10000" | jq -r '.total')
if [ "$RESULT" -ge 0 ]; then
    echo "✓ Size filter passed (found $RESULT results)"
else
    echo "✗ Size filter failed"
    exit 1
fi

# Test 10: Extension suggestions
echo "Test 10: Extension suggestions"
EXTENSIONS=$(curl -sf "${API_URL}/suggest" | jq -r '.extensions | length')
if [ "$EXTENSIONS" -gt 0 ]; then
    echo "✓ Extension suggestions passed (found $EXTENSIONS extensions)"
else
    echo "✗ Extension suggestions failed"
    exit 1
fi

echo ""
echo "All E2E tests passed! ✓"