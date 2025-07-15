
#!/bin/bash

# Test runner script for GW_predict_with_GNN
# This script runs all test files and provides detailed output

echo "==================================================================================="
echo "Running GW_predict_with_GNN Test Suite"
echo "Date: $(date)"
echo "==================================================================================="

# Function to run a test and check its result
run_test() {
    local test_file=$1
    local test_name=$2
    
    echo ""
    echo "Running $test_name tests..."
    echo "------------------------------------------------------------------------------------"
    
    if pytest "$test_file" -v -s; then
        echo "✓ $test_name tests PASSED"
    else
        echo "✗ $test_name tests FAILED"
        FAILED_TESTS+=("$test_name")
    fi
}

# Initialize array to track failed tests
FAILED_TESTS=()

# Run all tests - order matters: training must run before predictions
run_test "test_train_model.py" "Training Model"
run_test "test_predictions.py" "Predictions"
run_test "test_optimize_hyperparameters.py" "Hyperparameter Optimization"

echo ""
echo "==================================================================================="
echo "Test Suite Summary"
echo "==================================================================================="

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo "ALL TESTS PASSED!"
    echo "The GW_predict_with_GNN pipeline is working correctly."
    exit 0
else
    echo "Some tests failed:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "   - $test"
    done
    echo ""
    echo "Please check the output above for details on the failures."
    exit 1
fi
