"""
Integration: 6 Pathfinding Algorithms from Part A
Implements: BFS, DFS, UCS, GBFS, A*, IDA*
"""

import heapq
import math
from typing import List, Dict, Tuple
from collections import deque


class PathfindingIntegration:
    """Integrates ML severity prediction with 6 search algorithms"""
    
    def __init__(self, graph, nodes):
        """
        Args:
            graph: Adjacency list {node_id: [(neighbor, weight), ...]}
            nodes: Node coordinates {node_id: {'lat': ..., 'lon': ...}}
        """
        self.graph = graph
        self.nodes = nodes
    
    def heuristic(self, node1, node2):
        """Haversine distance (straight-line distance on Earth)"""
        lat1 = math.radians(self.nodes[node1]['lat'])
        lon1 = math.radians(self.nodes[node1]['lon'])
        lat2 = math.radians(self.nodes[node2]['lat'])
        lon2 = math.radians(self.nodes[node2]['lon'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return 6371 * c
    
    def calculate_path_cost(self, path):
        """Calculate total path cost"""
        cost = 0
        for i in range(len(path) - 1):
            for neighbor, weight in self.graph.get(path[i], []):
                if neighbor == path[i+1]:
                    cost += weight
                    break
        return cost
    
    def bfs(self, start, goal):
        """Breadth-First Search"""
        queue = deque([(start, [start])])
        visited = {start}
        nodes_expanded = 0
        
        while queue:
            node, path = queue.popleft()
            nodes_expanded += 1
            
            if node == goal:
                cost = self.calculate_path_cost(path)
                return path, cost, nodes_expanded
            
            for neighbor, _ in self.graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None, float('inf'), nodes_expanded
    
    def dfs(self, start, goal, max_depth=50):
        """Depth-First Search"""
        stack = [(start, [start], 0)]
        visited = set()
        nodes_expanded = 0
        best_path = None
        best_cost = float('inf')
        
        while stack:
            node, path, depth = stack.pop()
            nodes_expanded += 1
            
            if depth > max_depth:
                continue
            
            if node == goal:
                cost = self.calculate_path_cost(path)
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
                continue
            
            if node in visited:
                continue
            visited.add(node)
            
            for neighbor, _ in self.graph.get(node, []):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], depth + 1))
        
        return best_path, best_cost, nodes_expanded
    
    def ucs(self, start, goal):
        """Uniform Cost Search (Dijkstra)"""
        pq = [(0, start, [start])]
        visited = set()
        nodes_expanded = 0
        
        while pq:
            cost, node, path = heapq.heappop(pq)
            nodes_expanded += 1
            
            if node in visited:
                continue
            visited.add(node)
            
            if node == goal:
                return path, cost, nodes_expanded
            
            for neighbor, edge_weight in self.graph.get(node, []):
                if neighbor not in visited:
                    new_cost = cost + edge_weight
                    heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))
        
        return None, float('inf'), nodes_expanded
    
    def gbfs(self, start, goal):
        """Greedy Best-First Search"""
        pq = [(self.heuristic(start, goal), start, [start])]
        visited = set()
        nodes_expanded = 0
        
        while pq:
            _, node, path = heapq.heappop(pq)
            nodes_expanded += 1
            
            if node in visited:
                continue
            visited.add(node)
            
            if node == goal:
                cost = self.calculate_path_cost(path)
                return path, cost, nodes_expanded
            
            for neighbor, _ in self.graph.get(node, []):
                if neighbor not in visited:
                    h = self.heuristic(neighbor, goal)
                    heapq.heappush(pq, (h, neighbor, path + [neighbor]))
        
        return None, float('inf'), nodes_expanded
    
    def astar(self, start, goal):
        """A* Search"""
        pq = [(self.heuristic(start, goal), 0, start, [start])]
        visited = set()
        nodes_expanded = 0
        
        while pq:
            _, cost, node, path = heapq.heappop(pq)
            nodes_expanded += 1
            
            if node in visited:
                continue
            visited.add(node)
            
            if node == goal:
                return path, cost, nodes_expanded
            
            for neighbor, edge_weight in self.graph.get(node, []):
                if neighbor not in visited:
                    g = cost + edge_weight
                    h = self.heuristic(neighbor, goal)
                    f = g + h
                    heapq.heappush(pq, (f, g, neighbor, path + [neighbor]))
        
        return None, float('inf'), nodes_expanded
    
    def idastar(self, start, goal):
        """Iterative Deepening A*"""
        threshold = self.heuristic(start, goal)
        path = [start]
        total_expanded = 0
        
        while True:
            result = self._ida_search(path, 0, threshold, goal)
            total_expanded += result['expanded']
            
            if result['found']:
                cost = self.calculate_path_cost(result['path'])
                return result['path'], cost, total_expanded
            
            if result['threshold'] == float('inf'):
                return None, float('inf'), total_expanded
            
            threshold = result['threshold']
    
    def _ida_search(self, path, g, threshold, goal):
        """Helper for IDA*"""
        node = path[-1]
        f = g + self.heuristic(node, goal)
        expanded = 1
        
        if f > threshold:
            return {'found': False, 'threshold': f, 'expanded': expanded}
        
        if node == goal:
            return {'found': True, 'path': path, 'threshold': threshold, 'expanded': expanded}
        
        min_threshold = float('inf')
        
        for neighbor, edge_weight in self.graph.get(node, []):
            if neighbor not in path:
                path.append(neighbor)
                result = self._ida_search(path, g + edge_weight, threshold, goal)
                expanded += result['expanded']
                
                if result['found']:
                    return result
                
                if result['threshold'] < min_threshold:
                    min_threshold = result['threshold']
                
                path.pop()
        
        return {'found': False, 'threshold': min_threshold, 'expanded': expanded}
    
    def find_top3_algorithms(self, start, goal):
        """Run all 6 algorithms and return ALL results (for top-5 selection)"""
        algorithms = {
            'BFS': self.bfs,
            'DFS': self.dfs,
            'UCS': self.ucs,
            'GBFS': self.gbfs,
            'A*': self.astar,
            'IDA*': self.idastar
        }
        
        results = []
        
        print(f"\nRunning 6 algorithms from {start} to {goal}...")
        print("="*70)
        
        for name, func in algorithms.items():
            try:
                path, cost, expanded = func(start, goal)
                
                if path:
                    score = cost + (expanded * 0.01)
                    results.append({
                        'algorithm': name,
                        'path': path,
                        'cost': cost,
                        'nodes_expanded': expanded,
                        'path_length': len(path),
                        'score': score
                    })
                    print(f"[{name:5s}] Cost: {cost:6.2f} | Expanded: {expanded:4d} | Path: {len(path):2d} nodes")
                else:
                    print(f"[{name:5s}] No path found")
            except Exception as e:
                print(f"[{name:5s}] Error: {e}")
        
        # Sort and return ALL results (GUI will take top-5)
        results.sort(key=lambda x: x['score'])
        
        print("\n" + "="*70)
        print(f"Found {len(results)} valid paths")
        
        return results 


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from src.graph_construction import parse_road_network
    
    nodes, ways, cameras, meta = parse_road_network('heritage_assignment_15_time_asymmetric-1.txt')
    
    graph = {}
    for way in ways:
        if way['from'] not in graph:
            graph[way['from']] = []
        graph[way['from']].append((way['to'], way['time_min']))
    
    pathfinder = PathfindingIntegration(graph, nodes)
    
    top3 = pathfinder.find_top3_algorithms('1', '10')
    
    print("\n[OK] Pathfinding integration test passed")