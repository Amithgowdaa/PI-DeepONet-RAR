import os
import sys
import importlib.util
import traceback

def run_all_tests():
    print("========================================")
    print("Starting PI-DeepONet-RAR Test Suite")
    print("========================================")
    
    tests_dir = os.path.join(os.path.dirname(__file__), 'tests')
    if not os.path.exists(tests_dir):
        print(f"Error: tests directory not found at {tests_dir}")
        sys.exit(1)
        
    test_files = [f for f in os.listdir(tests_dir) if f.startswith('test_') and f.endswith('.py')]
    
    passed_count = 0
    failed_count = 0
    
    for file in sorted(test_files):
        print(f"\nRunning tests in {file}:")
        file_path = os.path.join(tests_dir, file)
        
        # Load module dynamically
        module_name = file[:-3]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"  [ERROR] Failed to load {file}: {e}")
            traceback.print_exc()
            failed_count += 1
            continue
            
        # Find all test functions
        test_functions = [getattr(module, name) for name in dir(module) 
                          if name.startswith('test_') and callable(getattr(module, name))]
        
        for func in test_functions:
            func_name = func.__name__
            try:
                func()
                print(f"  [PASS] {func_name}")
                passed_count += 1
            except AssertionError as ae:
                print(f"  [FAIL] {func_name}: Assertion Error")
                print(f"    Details: {ae}")
                failed_count += 1
            except Exception as e:
                print(f"  [FAIL] {func_name}: Unexpected error")
                traceback.print_exc(limit=3)
                failed_count += 1
                
    print("\n========================================")
    print(f"Test Summary: {passed_count} passed, {failed_count} failed")
    print("========================================")
    
    if failed_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    run_all_tests()
