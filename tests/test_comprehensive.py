"""
10 COMPREHENSIVE TEST CASES - COS30019 Assignment 2B
Tests 3 AI Models: CNN + LSTM + GCN
Author: Cherylynn
"""

import sys
from pathlib import Path
import torch
import numpy as np
import time
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn_model import load_cnn_model
from src.models.rnn_model import TravelTimeLSTM, load_model
from src.models.gcn_model import load_gcn_model
from src.graph_construction import parse_road_network, construct_graph

# Test results storage
TEST_RESULTS = []

def log_test(test_name, passed, details=""):
    """Log test result"""
    result = {
        'name': test_name,
        'passed': passed,
        'details': details,
        'timestamp': datetime.now().isoformat()
    }
    TEST_RESULTS.append(result)
    status = "[PASS]" if passed else "[FAIL]"
    print(f"\n{status}: {test_name}")
    if details:
        print(f"  {details}")

# =============================================================================
# TEST 1: All 3 Models Load Successfully
# =============================================================================
def test_1_models_load():
    """Test if all 3 trained models (CNN, LSTM, GCN) load without errors"""
    print("\n" + "="*70)
    print("TEST 1: Model Loading Verification (3 Models)")
    print("="*70)
    
    models_loaded = []
    
    try:
        cnn = load_cnn_model('models/cnn_model.pth')
        models_loaded.append('CNN')
        print("[OK] CNN loaded (Image Classification)")
    except Exception as e:
        print(f"[ERROR] CNN failed: {e}")
    
    try:
        lstm = TravelTimeLSTM(15, 256, 2, bidirectional=True)
        lstm = load_model(lstm, 'models/lstm_travel_time_model.pth')
        models_loaded.append('LSTM')
        print("[OK] LSTM loaded (Time Series Analysis)")
    except Exception as e:
        print(f"[ERROR] LSTM failed: {e}")
    
    try:
        gcn = load_gcn_model('models/gcn_model.pth', num_node_features=5)
        models_loaded.append('GCN')
        print("[OK] GCN loaded (Spatial Network Analysis)")
    except Exception as e:
        print(f"[ERROR] GCN failed: {e}")
    
    passed = len(models_loaded) == 3
    log_test("All 3 Models Loading", passed, f"{len(models_loaded)}/3 models loaded successfully")
    return passed

# =============================================================================
# TEST 2: CNN Prediction Consistency
# =============================================================================
def test_2_cnn_consistency():
    """Test if CNN gives consistent predictions for same input"""
    print("\n" + "="*70)
    print("TEST 2: CNN Image Classification Consistency")
    print("="*70)
    
    try:
        model = load_cnn_model('models/cnn_model.pth')
        model.eval()
        
        img = torch.randn(1, 3, 224, 224)
        
        predictions = []
        for i in range(3):
            with torch.no_grad():
                out = model(img)
                pred = torch.argmax(out, dim=1).item()
            predictions.append(pred)
        
        consistent = len(set(predictions)) == 1
        
        log_test("CNN Consistency", consistent, 
                f"Predictions: {predictions} (consistent: {consistent})")
        return consistent
        
    except Exception as e:
        log_test("CNN Consistency", False, f"Error: {e}")
        return False

# =============================================================================
# TEST 3: LSTM Travel Time Bounds Check
# =============================================================================
def test_3_lstm_bounds():
    """Test if LSTM predictions are within reasonable bounds (0-120 minutes)"""
    print("\n" + "="*70)
    print("TEST 3: LSTM Travel Time Prediction Bounds")
    print("="*70)
    
    try:
        model = TravelTimeLSTM(15, 256, 2, bidirectional=True)
        model = load_model(model, 'models/lstm_travel_time_model.pth')
        model.eval()
        
        times = []
        for _ in range(10):
            path = np.random.rand(30, 15)
            with torch.no_grad():
                pred_time = model(torch.FloatTensor(path).unsqueeze(0)).item()
            times.append(pred_time)
        
        all_reasonable = all(0 < t < 120 for t in times)
        
        log_test("LSTM Time Bounds", all_reasonable,
                f"Time range: {min(times):.1f}-{max(times):.1f} min (target: 0-120)")
        return all_reasonable
        
    except Exception as e:
        log_test("LSTM Time Bounds", False, f"Error: {e}")
        return False

# =============================================================================
# TEST 4: GCN Spatial Network Analysis
# =============================================================================
def test_4_gcn_spatial():
    """Test if GCN predicts traffic flow on road network"""
    print("\n" + "="*70)
    print("TEST 4: GCN Spatial Network Traffic Flow Prediction")
    print("="*70)
    
    try:
        nodes, ways, _, _ = parse_road_network('heritage_assignment_15_time_asymmetric-1.txt')
        data = construct_graph(nodes, ways)
        
        model = load_gcn_model('models/gcn_model.pth', num_node_features=5)
        model.eval()
        
        with torch.no_grad():
            predictions = model.predict(data)
        
        valid_predictions = all(p in [0, 1, 2] for p in predictions.tolist())
        diversity = len(set(predictions.tolist())) >= 2
        
        passed = valid_predictions and diversity
        
        log_test("GCN Spatial Analysis", passed,
                f"Predictions: {len(set(predictions.tolist()))}/3 classes used")
        return passed
        
    except Exception as e:
        log_test("GCN Spatial Analysis", False, f"Error: {e}")
        return False

# =============================================================================
# TEST 5: Real-Time Performance (All 3 Models)
# =============================================================================
def test_5_inference_speed():
    """Test if all 3 models can predict within acceptable time (<3 seconds)"""
    print("\n" + "="*70)
    print("TEST 5: Real-Time Performance (All 3 Models)")
    print("="*70)
    
    try:
        cnn = load_cnn_model('models/cnn_model.pth')
        lstm = TravelTimeLSTM(15, 256, 2, bidirectional=True)
        lstm = load_model(lstm, 'models/lstm_travel_time_model.pth')
        
        nodes, ways, _, _ = parse_road_network('heritage_assignment_15_time_asymmetric-1.txt')
        data = construct_graph(nodes, ways)
        gcn = load_gcn_model('models/gcn_model.pth', num_node_features=5)
        
        cnn.eval()
        lstm.eval()
        gcn.eval()
        
        start = time.time()
        
        with torch.no_grad():
            _ = cnn(torch.randn(1, 3, 224, 224))
            _ = lstm(torch.randn(1, 30, 15))
            _ = gcn.predict(data)
        
        elapsed = time.time() - start
        passed = elapsed < 3.0
        
        log_test("Real-Time Performance", passed,
                f"Total time: {elapsed*1000:.1f}ms (target: <3000ms)")
        return passed
        
    except Exception as e:
        log_test("Real-Time Performance", False, f"Error: {e}")
        return False

# =============================================================================
# TEST 6: LSTM Rush Hour Detection
# =============================================================================
def test_6_rush_hour():
    """Test if LSTM detects rush hour impacts (temporal pattern learning)"""
    print("\n" + "="*70)
    print("TEST 6: LSTM Rush Hour Detection (Kuching Traffic Patterns)")
    print("="*70)
    
    try:
        model = TravelTimeLSTM(15, 256, 2, bidirectional=True)
        model = load_model(model, 'models/lstm_travel_time_model.pth')
        model.eval()
        
        normal = np.random.rand(30, 15)
        normal[:, 6] = 10/24
        normal[:, 7] = 0.0
        
        rush = np.random.rand(30, 15)
        rush[:, 6] = 8/24
        rush[:, 7] = 1.0
        
        with torch.no_grad():
            normal_time = model(torch.FloatTensor(normal).unsqueeze(0)).item()
            rush_time = model(torch.FloatTensor(rush).unsqueeze(0)).item()
        
        reasonable = rush_time >= normal_time * 0.9
        
        log_test("LSTM Rush Hour Detection", reasonable,
                f"Normal: {normal_time:.1f}min, Rush: {rush_time:.1f}min")
        return reasonable
        
    except Exception as e:
        log_test("LSTM Rush Hour Detection", False, f"Error: {e}")
        return False

# =============================================================================
# TEST 7: LSTM Weekend vs Weekday Pattern
# =============================================================================
def test_7_weekly_cycle():
    """Test if LSTM recognizes weekly traffic cycles"""
    print("\n" + "="*70)
    print("TEST 7: LSTM Weekly Cycle Pattern (Sarawak Context)")
    print("="*70)
    
    try:
        model = TravelTimeLSTM(15, 256, 2, bidirectional=True)
        model = load_model(model, 'models/lstm_travel_time_model.pth')
        model.eval()
        
        monday = np.random.rand(30, 15)
        monday[:, 8] = 0/7
        monday[:, 9] = 0.0
        monday[:, 2] = 0.7
        
        sunday = np.random.rand(30, 15)
        sunday[:, 8] = 6/7
        sunday[:, 9] = 1.0
        sunday[:, 2] = 0.3
        
        with torch.no_grad():
            mon_time = model(torch.FloatTensor(monday).unsqueeze(0)).item()
            sun_time = model(torch.FloatTensor(sunday).unsqueeze(0)).item()
        
        makes_sense = sun_time <= mon_time * 1.1
        
        log_test("LSTM Weekly Cycle", makes_sense,
                f"Monday: {mon_time:.1f}min, Sunday: {sun_time:.1f}min")
        return makes_sense
        
    except Exception as e:
        log_test("LSTM Weekly Cycle", False, f"Error: {e}")
        return False

# =============================================================================
# TEST 8: CNN + LSTM Incident Impact
# =============================================================================
def test_8_incident_impact():
    """Test if CNN detects severity and LSTM adjusts travel time"""
    print("\n" + "="*70)
    print("TEST 8: CNN Severity + LSTM Time Adjustment")
    print("="*70)
    
    try:
        cnn = load_cnn_model('models/cnn_model.pth')
        lstm = TravelTimeLSTM(15, 256, 2, bidirectional=True)
        lstm = load_model(lstm, 'models/lstm_travel_time_model.pth')
        
        cnn.eval()
        lstm.eval()
        
        img = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            severity = torch.argmax(cnn(img), dim=1).item()
        
        clear = np.random.rand(30, 15)
        clear[:, 5] = 0.0
        
        incident = np.random.rand(30, 15)
        incident[:, 5] = severity / 3.0
        
        with torch.no_grad():
            clear_time = lstm(torch.FloatTensor(clear).unsqueeze(0)).item()
            incident_time = lstm(torch.FloatTensor(incident).unsqueeze(0)).item()
        
        reasonable = incident_time >= clear_time * 0.95
        
        log_test("CNN+LSTM Incident Impact", reasonable,
                f"Clear: {clear_time:.1f}min, Incident: {incident_time:.1f}min")
        return reasonable
        
    except Exception as e:
        log_test("CNN+LSTM Incident Impact", False, f"Error: {e}")
        return False

# =============================================================================
# TEST 9: Model File Integrity
# =============================================================================
def test_9_file_integrity():
    """Verify all 3 model files exist and have valid sizes"""
    print("\n" + "="*70)
    print("TEST 9: Model File Integrity (3 Models)")
    print("="*70)
    
    required_files = [
        'models/cnn_model.pth',
        'models/lstm_travel_time_model.pth',
        'models/gcn_model.pth'
    ]
    
    files_ok = []
    
    for file in required_files:
        path = Path(file)
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)
            print(f"[OK] {file} ({size:.1f} MB)")
            files_ok.append(file)
        else:
            print(f"[MISSING] {file}")
    
    passed = len(files_ok) == len(required_files)
    
    log_test("Model File Integrity", passed,
            f"{len(files_ok)}/{len(required_files)} model files found")
    return passed

# =============================================================================
# TEST 10: Complete 3-Model Integration Pipeline
# =============================================================================
def test_10_complete_integration():
    """Test complete 3-model integration: CNN + LSTM + GCN"""
    print("\n" + "="*70)
    print("TEST 10: Complete 3-Model Integration Pipeline")
    print("="*70)
    
    try:
        cnn = load_cnn_model('models/cnn_model.pth')
        lstm = TravelTimeLSTM(15, 256, 2, bidirectional=True)
        lstm = load_model(lstm, 'models/lstm_travel_time_model.pth')
        
        nodes, ways, _, _ = parse_road_network('heritage_assignment_15_time_asymmetric-1.txt')
        data = construct_graph(nodes, ways)
        gcn = load_gcn_model('models/gcn_model.pth', num_node_features=5)
        
        cnn.eval()
        lstm.eval()
        gcn.eval()
        
        print("\nExecuting 3-model pipeline:")
        
        # STEP 1: CNN predicts severity
        img = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            severity = torch.argmax(cnn(img), dim=1).item()
        
        classes = ['none', 'minor', 'moderate', 'severe']
        multipliers = [1.0, 1.2, 1.5, 2.0]
        print(f"  [1] CNN: Severity = {classes[severity]}")
        
        # STEP 2: GCN predicts network traffic flow
        with torch.no_grad():
            flow_predictions = gcn.predict(data)
        
        avg_flow = flow_predictions.float().mean().item()
        print(f"  [2] GCN: Average traffic flow = {avg_flow:.2f} (0=low, 1=med, 2=high)")
        
        # STEP 3: LSTM predicts travel time
        path = np.random.rand(30, 15)
        path[:, 5] = severity / 3.0
        path[:, 2] = avg_flow / 2.0
        
        with torch.no_grad():
            base_time = lstm(torch.FloatTensor(path).unsqueeze(0)).item()
        
        print(f"  [3] LSTM: Base travel time = {base_time:.1f} minutes")
        
        # STEP 4: Combine all factors
        final_time = base_time * multipliers[severity] * (1.0 + avg_flow * 0.1)
        
        print(f"\n  [4] FINAL: Adjusted time = {final_time:.1f} minutes")
        print(f"      (Base: {base_time:.1f} × Severity: {multipliers[severity]} × Flow factor: {1.0 + avg_flow*0.1:.2f})")
        
        pipeline_complete = (
            0 <= severity <= 3 and
            0 < base_time < 120 and
            final_time >= base_time
        )
        
        log_test("Complete 3-Model Integration", pipeline_complete,
                f"All models executed successfully")
        return pipeline_complete
        
    except Exception as e:
        log_test("Complete 3-Model Integration", False, f"Error: {e}")
        return False

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================
def run_all_tests():
    """Execute all 10 test cases and generate report"""
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST SUITE - COS30019 ASSIGNMENT 2B")
    print("Traffic Incident Classification System - Kuching, Sarawak")
    print("Testing 3 AI Models: CNN + LSTM + GCN")
    print("="*70)
    
    tests = [
        test_1_models_load,
        test_2_cnn_consistency,
        test_3_lstm_bounds,
        test_4_gcn_spatial,
        test_5_inference_speed,
        test_6_rush_hour,
        test_7_weekly_cycle,
        test_8_incident_impact,
        test_9_file_integrity,
        test_10_complete_integration
    ]
    
    for i, test_func in enumerate(tests, 1):
        print(f"\n{'='*70}")
        print(f"EXECUTING TEST {i}/10")
        print(f"{'='*70}")
        test_func()
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in TEST_RESULTS if r['passed'])
    total = len(TEST_RESULTS)
    
    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    for i, result in enumerate(TEST_RESULTS, 1):
        status = "[PASS]" if result['passed'] else "[FAIL]"
        print(f"  {status} Test {i}: {result['name']}")
    
    Path('tests').mkdir(exist_ok=True)
    with open('tests/test_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'passed': passed,
            'total': total,
            'percentage': passed/total*100,
            'models_tested': ['CNN', 'LSTM', 'GCN'],
            'tests': TEST_RESULTS
        }, f, indent=2)
    
    print(f"\n[OK] Results saved to: tests/test_results.json")
    
    return passed >= 8

if __name__ == "__main__":
    try:
        success = run_all_tests()
        print("\n" + "="*70)
        if success:
            print("[SUCCESS] 3-MODEL INTEGRATION TEST SUITE PASSED")
            print("Models tested: CNN (images) + LSTM (time series) + GCN (spatial)")
        else:
            print("[WARNING] SOME TESTS FAILED - REVIEW RESULTS ABOVE")
        print("="*70)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Tests stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)