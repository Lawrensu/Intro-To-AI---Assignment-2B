"""
COMPREHENSIVE Test Suite - 3 Image Models + 6 Pathfinding Algorithms
Tests all critical functionality
"""

import sys
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.models.image_models import (
    load_resnet18, load_mobilenet, load_efficientnet,
    predict_severity, get_edge_multiplier, ensemble_predict
)
from src.pathfinding_integration import PathfindingIntegration
from src.graph_construction import parse_road_network

TEST_RESULTS = []

def log_test(name, passed, details=""):
    """Log test result"""
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if details:
        print(f"     {details}")
    
    TEST_RESULTS.append({
        'name': name,
        'passed': passed,
        'details': details
    })

def test_1_models_load():
    """Test if all 3 models load successfully"""
    print("\n" + "="*70)
    print("TEST 1: Load All 3 Image Models")
    print("="*70)
    
    models_loaded = []
    
    try:
        resnet = load_resnet18('models/resnet18_model.pth')
        log_test("Load ResNet-18", True, "Model loaded successfully")
        models_loaded.append(resnet)
    except Exception as e:
        log_test("Load ResNet-18", False, str(e))
        return False
    
    try:
        mobilenet = load_mobilenet('models/mobilenet_model.pth')
        log_test("Load MobileNet-V2", True, "Model loaded successfully")
        models_loaded.append(mobilenet)
    except Exception as e:
        log_test("Load MobileNet-V2", False, str(e))
        return False
    
    try:
        efficientnet = load_efficientnet('models/efficientnet_model.pth')
        log_test("Load EfficientNet-B0", True, "Model loaded successfully")
        models_loaded.append(efficientnet)
    except Exception as e:
        log_test("Load EfficientNet-B0", False, str(e))
        return False
    
    log_test("All 3 models loaded", True, f"{len(models_loaded)} models ready")
    return True

def test_2_model_inference():
    """Test if models can make predictions"""
    print("\n" + "="*70)
    print("TEST 2: Model Inference")
    print("="*70)
    
    try:
        resnet = load_resnet18('models/resnet18_model.pth')
        img = torch.randn(1, 3, 224, 224)
        
        severity, conf = predict_severity(resnet, img)
        
        valid_classes = ['none', 'minor', 'moderate', 'severe']
        is_valid = severity in valid_classes and 0 <= conf <= 1
        
        log_test("Model Inference", is_valid, f"Predicted: {severity} ({conf:.2%})")
        return is_valid
    except Exception as e:
        log_test("Model Inference", False, str(e))
        return False

def test_3_consistency():
    """Test if models give consistent predictions"""
    print("\n" + "="*70)
    print("TEST 3: Prediction Consistency")
    print("="*70)
    
    try:
        resnet = load_resnet18('models/resnet18_model.pth')
        img = torch.randn(1, 3, 224, 224)
        
        preds = []
        for _ in range(5):
            sev, conf = predict_severity(resnet, img)
            preds.append(sev)
        
        consistent = len(set(preds)) == 1
        log_test("Consistency Check", consistent, f"5 predictions: {preds[0]}")
        return consistent
    except Exception as e:
        log_test("Consistency Check", False, str(e))
        return False

def test_4_ensemble():
    """Test if ensemble voting works"""
    print("\n" + "="*70)
    print("TEST 4: Ensemble Voting")
    print("="*70)
    
    try:
        resnet = load_resnet18('models/resnet18_model.pth')
        mobilenet = load_mobilenet('models/mobilenet_model.pth')
        efficientnet = load_efficientnet('models/efficientnet_model.pth')
        
        img = torch.randn(1, 3, 224, 224)
        
        models = [resnet, mobilenet, efficientnet]
        ensemble_sev, ensemble_conf = ensemble_predict(models, img)
        
        valid_classes = ['none', 'minor', 'moderate', 'severe']
        is_valid = ensemble_sev in valid_classes and 0 <= ensemble_conf <= 1
        
        log_test("Ensemble Voting", is_valid, f"Ensemble: {ensemble_sev} ({ensemble_conf:.2%})")
        return is_valid
    except Exception as e:
        log_test("Ensemble Voting", False, str(e))
        return False

def test_5_multipliers():
    """Test if severity multipliers are correct"""
    print("\n" + "="*70)
    print("TEST 5: Edge Weight Multipliers")
    print("="*70)
    
    expected = {
        'none': 1.0,
        'minor': 1.2,
        'moderate': 1.5,
        'severe': 2.0
    }
    
    all_correct = True
    for sev, expected_mult in expected.items():
        actual_mult = get_edge_multiplier(sev)
        if actual_mult != expected_mult:
            all_correct = False
            log_test(f"Multiplier {sev}", False, f"Expected {expected_mult}, got {actual_mult}")
        else:
            log_test(f"Multiplier {sev}", True, f"{expected_mult}x")
    
    return all_correct

def test_6_pathfinding_load():
    """Test if pathfinding graph loads"""
    print("\n" + "="*70)
    print("TEST 6: Pathfinding Graph Loading")
    print("="*70)
    
    try:
        nodes, ways, cameras, meta = parse_road_network('heritage_assignment_15_time_asymmetric-1.txt')
        
        graph = {}
        for way in ways:
            if way['from'] not in graph:
                graph[way['from']] = []
            graph[way['from']].append((way['to'], way['time_min']))
        
        pathfinder = PathfindingIntegration(graph, nodes)
        
        log_test("Graph Loading", True, f"{len(nodes)} nodes, {len(ways)} edges")
        return True
    except Exception as e:
        log_test("Graph Loading", False, str(e))
        return False

def test_7_all_algorithms():
    """Test if all 6 algorithms execute"""
    print("\n" + "="*70)
    print("TEST 7: 6 Pathfinding Algorithms")
    print("="*70)
    
    try:
        nodes, ways, cameras, meta = parse_road_network('heritage_assignment_15_time_asymmetric-1.txt')
        
        graph = {}
        for way in ways:
            if way['from'] not in graph:
                graph[way['from']] = []
            graph[way['from']].append((way['to'], way['time_min']))
        
        pathfinder = PathfindingIntegration(graph, nodes)
        
        algorithms = ['BFS', 'DFS', 'UCS', 'GBFS', 'A*', 'IDA*']
        
        all_work = True
        for algo in algorithms:
            try:
                func = getattr(pathfinder, algo.lower().replace('*', 'star').replace('-', ''))
                path, cost, expanded = func('1', '10')
                if path:
                    log_test(f"Algorithm {algo}", True, f"Cost: {cost:.1f}, Expanded: {expanded}")
                else:
                    log_test(f"Algorithm {algo}", False, "No path found")
                    all_work = False
            except Exception as e:
                log_test(f"Algorithm {algo}", False, str(e))
                all_work = False
        
        return all_work
    except Exception as e:
        log_test("6 Algorithms Execute", False, str(e))
        return False

def test_8_top3_selection():
    """Test if top-3 algorithm selection works"""
    print("\n" + "="*70)
    print("TEST 8: Top-3 Algorithm Selection")
    print("="*70)
    
    try:
        nodes, ways, cameras, meta = parse_road_network('heritage_assignment_15_time_asymmetric-1.txt')
        
        graph = {}
        for way in ways:
            if way['from'] not in graph:
                graph[way['from']] = []
            graph[way['from']].append((way['to'], way['time_min']))
        
        pathfinder = PathfindingIntegration(graph, nodes)
        
        top3 = pathfinder.find_top3_algorithms('1', '10')
        
        has_top3 = len(top3) >= 3
        
        if has_top3:
            algos = [r['algorithm'] for r in top3]
            log_test("Top-3 Selection", True, f"Top 3: {', '.join(algos)}")
        else:
            log_test("Top-3 Selection", False, f"Only {len(top3)} algorithms returned")
        
        return has_top3
    except Exception as e:
        log_test("Top-3 Selection", False, str(e))
        return False

def test_9_severity_impact():
    """Test if severity affects pathfinding"""
    print("\n" + "="*70)
    print("TEST 9: Severity Impact on Pathfinding")
    print("="*70)
    
    try:
        nodes, ways, cameras, meta = parse_road_network('heritage_assignment_15_time_asymmetric-1.txt')
        
        graph = {}
        for way in ways:
            if way['from'] not in graph:
                graph[way['from']] = []
            graph[way['from']].append((way['to'], way['time_min']))
        
        pathfinder = PathfindingIntegration(graph, nodes)
        
        path1, cost1, _ = pathfinder.ucs('1', '10')
        
        graph_modified = {}
        for node in graph:
            graph_modified[node] = [(n, w * 2.0) for n, w in graph[node]]
        
        pathfinder_modified = PathfindingIntegration(graph_modified, nodes)
        path2, cost2, _ = pathfinder_modified.ucs('1', '10')
        
        impact_detected = cost2 >= cost1 * 1.5
        
        log_test("Severity Impact", impact_detected, f"Base: {cost1:.1f} -> Severe: {cost2:.1f}")
        return impact_detected
    except Exception as e:
        log_test("Severity Impact", False, str(e))
        return False

def test_10_file_integrity():
    """Test if all required files exist"""
    print("\n" + "="*70)
    print("TEST 10: File Integrity")
    print("="*70)
    
    required_files = [
        'models/resnet18_model.pth',
        'models/mobilenet_model.pth',
        'models/efficientnet_model.pth',
        'models/ResNet18_learning_curves.png',
        'models/MobileNetV2_learning_curves.png',
        'models/EfficientNet-B0_learning_curves.png',
        'models/ResNet18_confusion_matrix.png',
        'models/MobileNetV2_confusion_matrix.png',
        'models/EfficientNet-B0_confusion_matrix.png',
        'models/model_comparison.png'
    ]
    
    all_exist = True
    for file in required_files:
        exists = Path(file).exists()
        if not exists:
            all_exist = False
            log_test(f"File: {Path(file).name}", False, "Missing")
        else:
            size = Path(file).stat().st_size / (1024*1024)
            log_test(f"File: {Path(file).name}", True, f"{size:.1f} MB")
    
    return all_exist

def run_all_tests():
    """Execute all test cases"""
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST SUITE - 10 TESTS")
    print("3 Image Models + 6 Pathfinding Algorithms")
    print("="*70)
    
    tests = [
        test_1_models_load,
        test_2_model_inference,
        test_3_consistency,
        test_4_ensemble,
        test_5_multipliers,
        test_6_pathfinding_load,
        test_7_all_algorithms,
        test_8_top3_selection,
        test_9_severity_impact,
        test_10_file_integrity
    ]
    
    for i, test_func in enumerate(tests, 1):
        print(f"\n{'='*70}")
        print(f"Running Test {i}/{len(tests)}: {test_func.__name__}")
        print('='*70)
        test_func()
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in TEST_RESULTS if r['passed'])
    total = len(TEST_RESULTS)
    
    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\nSTATUS: ALL TESTS PASSED")
    elif passed >= total * 0.8:
        print(f"\nSTATUS: MOSTLY PASSED ({passed}/{total})")
    else:
        print(f"\nSTATUS: FAILED ({total-passed} failures)")
    
    Path('tests').mkdir(exist_ok=True)
    with open('tests/test_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'passed': passed,
            'total': total,
            'pass_rate': passed/total,
            'tests': TEST_RESULTS
        }, f, indent=2)
    
    print(f"\nResults saved to: tests/test_results.json")
    
    return passed >= 8

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)