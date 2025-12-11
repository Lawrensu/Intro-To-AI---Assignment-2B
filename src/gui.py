"""
Modern GUI Application - COS30019 Assignment 2B
Integrates CNN + LSTM + GCN with Part A pathfinding
Author: Team (Lawrence, Faridz, Cherylynn, Jason)
"""

import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import numpy as np
import webbrowser
from typing import Dict, List, Tuple
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn_model import load_cnn_model
from src.models.rnn_model import TravelTimeLSTM, load_model
from src.models.gcn_model import load_gcn_model
from src.graph_construction import parse_road_network, construct_graph

SEVERITY_MULTIPLIERS = {
    'none': 1.0,
    'minor': 1.2,
    'moderate': 1.5,
    'severe': 2.0
}

class ModernButton(tk.Canvas):
    """Custom modern button with hover effects and rounded corners"""
    def __init__(self, parent, text, command, bg_color, hover_color, text_color="white", **kwargs):
        super().__init__(parent, highlightthickness=0, **kwargs)
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.text = text
        
        width = kwargs.get('width', 200)
        height = kwargs.get('height', 50)
        
        self.config(width=width, height=height, bg=parent['bg'])
        
        # Create rounded rectangle
        radius = 8
        self.rect = self.create_rounded_rect(2, 2, width-2, height-2, radius, fill=bg_color)
        
        self.text_id = self.create_text(
            width/2, height/2,
            text=text,
            fill=text_color,
            font=("Segoe UI", 11, "bold")
        )
        
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        
    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        """Create a rounded rectangle"""
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
        
    def on_enter(self, e):
        self.itemconfig(self.rect, fill=self.hover_color)
        
    def on_leave(self, e):
        self.itemconfig(self.rect, fill=self.bg_color)
        
    def on_click(self, e):
        if self.command:
            self.command()


class RoundedFrame(tk.Canvas):
    """Custom frame with rounded corners"""
    def __init__(self, parent, radius=15, bg_color='#2d2d2d', **kwargs):
        tk.Canvas.__init__(self, parent, highlightthickness=0, **kwargs)
        self.bg_color = bg_color
        self.radius = radius
        
        self.config(bg=parent['bg'])
        
    def draw_rounded_rect(self):
        """Draw rounded rectangle background"""
        w = self.winfo_width()
        h = self.winfo_height()
        r = self.radius
        
        points = [
            r, 0,
            w-r, 0,
            w, 0,
            w, r,
            w, h-r,
            w, h,
            w-r, h,
            r, h,
            0, h,
            0, h-r,
            0, r,
            0, 0
        ]
        
        self.create_polygon(points, smooth=True, fill=self.bg_color, outline='')


class TrafficICS_GUI:
    """Traffic Incident Classification System - Modern GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic ICS - Kuching Heritage")
        self.root.geometry("1400x900")
        
        # Softer dark color scheme
        self.colors = {
            'bg': '#1e1e1e',           # Softer black background
            'card': '#2d2d2d',         # Card background
            'card_dark': '#252525',    # Darker card sections
            'accent': '#3a3a3a',       # Accent areas
            'primary': '#e94560',      # Primary action
            'success': '#10b981',      # Success green
            'warning': '#f59e0b',      # Warning orange
            'info': '#3b82f6',         # Info blue
            'purple': '#8b5cf6',       # Purple accent
            'text': '#f5f5f5',         # Main text
            'text_secondary': '#9ca3af', # Secondary text
            'border': '#404040',       # Subtle borders
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # State variables
        self.selected_image_path = None
        self.predicted_severity = None
        self.origin_node = tk.StringVar(value="1")
        self.destination_node = tk.StringVar(value="10")
        
        # Configure styles
        self.setup_styles()
        
        # Load models
        self.load_models()
        
        # Load road network
        self.load_network()
        
        # Build GUI
        self.create_widgets()
        
    def setup_styles(self):
        """Setup ttk styles for modern look"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Combobox style
        style.configure(
            'Modern.TCombobox',
            fieldbackground=self.colors['card_dark'],
            background=self.colors['card'],
            foreground=self.colors['text'],
            arrowcolor=self.colors['text'],
            borderwidth=1,
            relief='flat'
        )
        
        style.map('Modern.TCombobox',
            fieldbackground=[('readonly', self.colors['card_dark'])],
            selectbackground=[('readonly', self.colors['accent'])],
            selectforeground=[('readonly', self.colors['text'])],
            bordercolor=[('focus', self.colors['info'])]
        )
        
    def load_models(self):
        """Load all trained models"""
        print("Loading models...")
        
        try:
            self.cnn_model = load_cnn_model('models/cnn_model.pth')
            self.cnn_model.eval()
            print("[OK] CNN loaded")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load CNN: {e}")
            sys.exit(1)
        
        try:
            self.lstm_model = TravelTimeLSTM(15, 256, 2, bidirectional=True)
            self.lstm_model = load_model(self.lstm_model, 'models/lstm_travel_time_model.pth')
            self.lstm_model.eval()
            print("[OK] LSTM loaded")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load LSTM: {e}")
            sys.exit(1)
        
        try:
            self.gcn_model = load_gcn_model('models/gcn_model.pth', num_node_features=5)
            self.gcn_model.eval()
            print("[OK] GCN loaded")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load GCN: {e}")
            sys.exit(1)
    
    def load_network(self):
        """Load road network data"""
        print("Loading road network...")
        
        try:
            nodes, ways, cameras, meta = parse_road_network('heritage_assignment_15_time_asymmetric-1.txt')
            self.network_data = construct_graph(nodes, ways)
            self.nodes = nodes
            self.ways = ways
            self.cameras = cameras
            print(f"[OK] Loaded {len(nodes)} nodes, {len(ways)} edges")
        except Exception as e:
            messagebox.showerror("Network Error", f"Failed to load network: {e}")
            sys.exit(1)
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Header with gradient effect
        header = tk.Frame(self.root, bg=self.colors['card'], height=100)
        header.pack(fill=tk.X, padx=0, pady=0)
        
        title_frame = tk.Frame(header, bg=self.colors['card'])
        title_frame.pack(expand=True, pady=15)
        
        tk.Label(
            title_frame,
            text="🚗 Traffic Incident Classification System",
            font=("Segoe UI", 26, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(pady=(0, 5))
        
        tk.Label(
            title_frame,
            text="AI-Powered Route Optimization for Kuching Heritage Area",
            font=("Segoe UI", 11),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        ).pack()
        
        # Main container with padding
        container = tk.Frame(self.root, bg=self.colors['bg'])
        container.pack(fill=tk.BOTH, expand=True, padx=25, pady=20)
        
        # Top section (Image + Route Selection)
        top_section = tk.Frame(container, bg=self.colors['bg'])
        top_section.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Left panel - Image upload
        self.create_left_panel(top_section)
        
        # Right panel - Route planning
        self.create_right_panel(top_section)
        
        # Bottom panel - Results
        self.create_bottom_panel(container)
    
    def create_card(self, parent, title, subtitle=""):
        """Create a modern card container with rounded corners"""
        # Outer frame with padding for shadow effect
        outer = tk.Frame(parent, bg=self.colors['bg'])
        
        # Inner card
        card = tk.Frame(outer, bg=self.colors['card'], relief=tk.FLAT, bd=0)
        card.pack(padx=2, pady=2, fill=tk.BOTH, expand=True)
        
        # Card header
        header = tk.Frame(card, bg=self.colors['card'])
        header.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        # Title with icon
        title_label = tk.Label(
            header,
            text=title,
            font=("Segoe UI", 15, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text'],
            anchor=tk.W
        )
        title_label.pack(side=tk.LEFT)
        
        if subtitle:
            subtitle_label = tk.Label(
                header,
                text=subtitle,
                font=("Segoe UI", 9),
                bg=self.colors['card'],
                fg=self.colors['text_secondary'],
                anchor=tk.E
            )
            subtitle_label.pack(side=tk.RIGHT)
        
        # Separator line
        separator = tk.Frame(card, bg=self.colors['border'], height=1)
        separator.pack(fill=tk.X, padx=20)
        
        # Card body
        body = tk.Frame(card, bg=self.colors['card'])
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        return outer, body
    
    def create_left_panel(self, parent):
        """Create left panel for image upload"""
        card, body = self.create_card(parent, "📸  Incident Image Analysis", "Step 1")
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Image display container
        image_container = tk.Frame(body, bg=self.colors['card_dark'], relief=tk.FLAT, bd=0)
        image_container.pack(pady=(10, 15), fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(
            image_container,
            text="No image selected\n\nClick 'Browse Image' to upload",
            bg=self.colors['card_dark'],
            fg=self.colors['text_secondary'],
            font=("Segoe UI", 10),
            anchor=tk.CENTER,
            justify=tk.CENTER
        )
        self.image_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Buttons container
        btn_container = tk.Frame(body, bg=self.colors['card'])
        btn_container.pack(pady=(5, 10))
        
        self.upload_btn = ModernButton(
            btn_container,
            "📁  Browse Image",
            self.upload_image,
            bg_color=self.colors['info'],
            hover_color='#2563eb',
            width=200,
            height=45
        )
        self.upload_btn.pack(pady=5)
        
        self.analyze_btn = ModernButton(
            btn_container,
            "🔍  Analyze Severity",
            self.analyze_image,
            bg_color=self.colors['primary'],
            hover_color='#dc2626',
            width=200,
            height=45
        )
        self.analyze_btn.pack(pady=5)
        self.analyze_btn.itemconfig(self.analyze_btn.rect, state='hidden')
        self.analyze_btn.itemconfig(self.analyze_btn.text_id, state='hidden')
        
        # Result display
        result_container = tk.Frame(body, bg=self.colors['card_dark'], relief=tk.FLAT, bd=0)
        result_container.pack(pady=(10, 0), fill=tk.X)
        
        self.severity_result = tk.Label(
            result_container,
            text="Severity: Not analyzed",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors['card_dark'],
            fg=self.colors['text_secondary'],
            pady=12
        )
        self.severity_result.pack()
    
    def create_right_panel(self, parent):
        """Create right panel for route planning"""
        card, body = self.create_card(parent, "🗺️  Route Planning", "Step 2")
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Origin selection
        origin_frame = tk.Frame(body, bg=self.colors['card'])
        origin_frame.pack(fill=tk.X, pady=(5, 10))
        
        tk.Label(
            origin_frame,
            text="Origin (Start Point)",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text'],
            anchor=tk.W
        ).pack(anchor=tk.W, pady=(0, 5))
        
        origin_dropdown = ttk.Combobox(
            origin_frame,
            textvariable=self.origin_node,
            values=self.get_node_labels(),
            state="readonly",
            font=("Segoe UI", 10),
            style='Modern.TCombobox'
        )
        origin_dropdown.pack(fill=tk.X, ipady=10)
        
        # Destination selection
        dest_frame = tk.Frame(body, bg=self.colors['card'])
        dest_frame.pack(fill=tk.X, pady=(10, 10))
        
        tk.Label(
            dest_frame,
            text="Destination (Goal)",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text'],
            anchor=tk.W
        ).pack(anchor=tk.W, pady=(0, 5))
        
        dest_dropdown = ttk.Combobox(
            dest_frame,
            textvariable=self.destination_node,
            values=self.get_node_labels(),
            state="readonly",
            font=("Segoe UI", 10),
            style='Modern.TCombobox'
        )
        dest_dropdown.pack(fill=tk.X, ipady=10)
        
        # Calculate button
        calc_container = tk.Frame(body, bg=self.colors['card'])
        calc_container.pack(pady=(20, 15))
        
        calc_btn = ModernButton(
            calc_container,
            "🚀  Calculate Optimal Route",
            self.calculate_route,
            bg_color=self.colors['success'],
            hover_color='#059669',
            width=280,
            height=50
        )
        calc_btn.pack()
        
        # Info box with gradient
        info_box = tk.Frame(body, bg=self.colors['card_dark'], relief=tk.FLAT, bd=0)
        info_box.pack(pady=(10, 0), fill=tk.X)
        
        tk.Label(
            info_box,
            text="AI Models Integration:",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['card_dark'],
            fg=self.colors['text'],
            anchor=tk.W
        ).pack(anchor=tk.W, padx=12, pady=(10, 5))
        
        models_text = [
            "• CNN - Image-based severity detection",
            "• GCN - Spatial traffic flow analysis",
            "• LSTM - Temporal travel prediction"
        ]
        
        for text in models_text:
            tk.Label(
                info_box,
                text=text,
                font=("Segoe UI", 9),
                bg=self.colors['card_dark'],
                fg=self.colors['text_secondary'],
                anchor=tk.W
            ).pack(anchor=tk.W, padx=12, pady=2)
        
        tk.Label(info_box, text="", bg=self.colors['card_dark']).pack(pady=5)
    
    def create_bottom_panel(self, parent):
        """Create bottom panel for results"""
        card, body = self.create_card(parent, "📊  Analysis Results", "Step 3")
        card.pack(fill=tk.BOTH, expand=True)
        
        # Results text area
        text_container = tk.Frame(body, bg=self.colors['card_dark'], relief=tk.FLAT, bd=0)
        text_container.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        self.results_text = tk.Text(
            text_container,
            height=10,
            font=("Consolas", 9),
            bg=self.colors['card_dark'],
            fg=self.colors['text'],
            wrap=tk.WORD,
            padx=12,
            pady=12,
            relief=tk.FLAT,
            insertbackground=self.colors['text'],
            selectbackground=self.colors['primary'],
            selectforeground=self.colors['text'],
            bd=0
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(self.results_text, bg=self.colors['card'])
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
        
        # Action buttons
        btn_frame = tk.Frame(body, bg=self.colors['card'])
        btn_frame.pack(pady=(10, 5))
        
        map_btn = ModernButton(
            btn_frame,
            "🗺️  View Map",
            self.view_map,
            bg_color=self.colors['purple'],
            hover_color='#7c3aed',
            width=140,
            height=40
        )
        map_btn.pack(side=tk.LEFT, padx=5)
        
        export_btn = ModernButton(
            btn_frame,
            "💾  Export",
            self.export_results,
            bg_color='#059669',
            hover_color='#047857',
            width=140,
            height=40
        )
        export_btn.pack(side=tk.LEFT, padx=5)
    
    def get_node_labels(self) -> List[str]:
        """Get formatted node labels for dropdowns"""
        labels = []
        for node_id, info in sorted(self.nodes.items()):
            labels.append(f"{node_id}: {info['label']}")
        return labels
    
    def upload_image(self):
        """Handle image upload"""
        file_path = filedialog.askopenfilename(
            title="Select Incident Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        
        if file_path:
            self.selected_image_path = file_path
            
            # Display image
            img = Image.open(file_path)
            img.thumbnail((350, 350))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
            # Show analyze button
            self.analyze_btn.itemconfig(self.analyze_btn.rect, state='normal')
            self.analyze_btn.itemconfig(self.analyze_btn.text_id, state='normal')
            
            self.log_result(f"✓ Image loaded: {Path(file_path).name}")
    
    def analyze_image(self):
        """Analyze uploaded image with CNN"""
        if not self.selected_image_path:
            messagebox.showwarning("No Image", "Please upload an image first")
            return
        
        self.log_result("\n[CNN] Analyzing image...")
        
        try:
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            img = Image.open(self.selected_image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                output = self.cnn_model(img_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                pred_class = torch.argmax(output, dim=1).item()
            
            classes = ['none', 'minor', 'moderate', 'severe']
            self.predicted_severity = classes[pred_class]
            confidence = probabilities[pred_class].item() * 100
            
            # Update UI with color coding
            severity_colors = {
                'none': self.colors['success'],
                'minor': self.colors['warning'],
                'moderate': '#fb923c',
                'severe': self.colors['primary']
            }
            
            severity_icons = {
                'none': '✓',
                'minor': '⚠',
                'moderate': '⚠⚠',
                'severe': '⛔'
            }
            
            self.severity_result.config(
                text=f"{severity_icons[self.predicted_severity]}  {self.predicted_severity.upper()}  ({confidence:.1f}% confidence)",
                fg=severity_colors[self.predicted_severity]
            )
            
            self.log_result(f"[CNN] Severity: {self.predicted_severity.upper()} ({confidence:.1f}% confidence)")
            self.log_result(f"[CNN] Time multiplier: {SEVERITY_MULTIPLIERS[self.predicted_severity]}x")
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze image: {e}")
            self.log_result(f"[ERROR] {e}")
    
    def calculate_route(self):
        """Calculate route with all 3 models and modified edge weights"""
        origin = self.origin_node.get().split(':')[0]
        destination = self.destination_node.get().split(':')[0]
        
        if origin == destination:
            messagebox.showwarning("Same Location", "Origin and destination cannot be the same")
            return
        
        self.results_text.delete(1.0, tk.END)
        self.log_result("="*70)
        self.log_result("ROUTE CALCULATION - 3-MODEL AI INTEGRATION")
        self.log_result("="*70)
        self.log_result(f"\n📍 Origin: {self.nodes[origin]['label']}")
        self.log_result(f"🎯 Destination: {self.nodes[destination]['label']}")
        
        try:
            # STEP 1: CNN predicts incident severity
            if self.predicted_severity:
                severity = self.predicted_severity
                multiplier = SEVERITY_MULTIPLIERS[severity]
                self.log_result(f"\n[1] CNN INCIDENT DETECTION")
                self.log_result(f"    Severity: {severity.upper()}")
                self.log_result(f"    Base Multiplier: {multiplier}x")
            else:
                severity = 'none'
                multiplier = 1.0
                self.log_result(f"\n[1] CNN INCIDENT DETECTION")
                self.log_result(f"    No image analyzed (using default)")
            
            # STEP 2: GCN predicts traffic flow at all nodes
            with torch.no_grad():
                flow_predictions = self.gcn_model.predict(self.network_data)
            
            avg_flow = flow_predictions.float().mean().item()
            flow_names = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
            self.log_result(f"\n[2] GCN NETWORK ANALYSIS")
            self.log_result(f"    Analyzing {len(flow_predictions)} intersections...")
            self.log_result(f"    Average Traffic Flow: {flow_names.get(int(avg_flow), 'UNKNOWN')}")
            
            # STEP 3: ⭐ MODIFY EDGE WEIGHTS ⭐
            from src.graph_construction import adjust_edge_weights_with_incident
            
            # Use origin as incident location (or could be set from image analysis)
            incident_node = origin
            
            self.log_result(f"\n[3] EDGE WEIGHT ADJUSTMENT")
            self.log_result(f"    Incident location: {self.nodes[incident_node]['label']}")
            
            adjusted_edges = adjust_edge_weights_with_incident(
                edges=self.ways,
                incident_node=incident_node,
                severity=severity,
                gcn_predictions=flow_predictions,
                node_to_idx=self.network_data.node_to_idx
            )
            
            self.log_result(f"    ✓ Updated {len(adjusted_edges)} road segments")
            
            # STEP 4: Run pathfinding with MODIFIED edges
            self.log_result(f"\n[4] PATHFINDING (Dijkstra with modified weights)")
            
            path, total_time = self.find_shortest_path_dijkstra(
                origin, 
                destination, 
                adjusted_edges
            )
            
            if not path:
                self.log_result(f"    ❌ No path found!")
                messagebox.showwarning("No Path", "No route found between selected locations")
                return
            
            self.log_result(f"    ✓ Path found: {len(path)} nodes, {len(path)-1} segments")
            self.log_result(f"    Pathfinding time: {total_time:.1f} minutes")
            
            # STEP 5: LSTM refines the travel time estimate
            path_features = self.create_path_features_from_path(path, adjusted_edges, severity)
            
            with torch.no_grad():
                lstm_time = self.lstm_model(torch.FloatTensor(path_features).unsqueeze(0)).item()
            
            self.log_result(f"\n[5] LSTM TIME REFINEMENT")
            self.log_result(f"    Pathfinding estimate: {total_time:.1f} min")
            self.log_result(f"    LSTM refined estimate: {lstm_time:.1f} min")
            self.log_result(f"    Difference: {abs(lstm_time - total_time):.1f} min")
            
            # Use LSTM estimate as final (it considers more factors)
            final_time = lstm_time
            
            # STEP 6: Display final results
            distance = self.calculate_path_distance(path, adjusted_edges)
            avg_speed = (distance / final_time * 60) if final_time > 0 else 0
            
            self.log_result(f"\n[6] FINAL ROUTE PLAN")
            self.log_result(f"    ───────────────────────────────")
            self.log_result(f"    Route: {' → '.join([self.nodes[n]['label'][:15] for n in path[:3]])}...")
            self.log_result(f"    Total nodes: {len(path)}")
            self.log_result(f"    Total segments: {len(path) - 1}")
            self.log_result(f"    ⏱️  Travel Time: {final_time:.1f} minutes")
            self.log_result(f"    📏 Distance: {distance:.2f} km")
            self.log_result(f"    🚗 Avg Speed: {avg_speed:.1f} km/h")
            
            # Show edge modifications (first few affected edges)
            self.log_result(f"\n[7] EDGE IMPACT ANALYSIS")
            affected_edges = [e for e in adjusted_edges if e.get('incident_factor', 1.0) > 1.0 or e.get('flow_factor', 1.0) > 1.0]
            
            if affected_edges:
                self.log_result(f"    Showing top 5 affected road segments:")
                for edge in affected_edges[:5]:
                    from_label = self.nodes[edge['from']]['label'][:20]
                    to_label = self.nodes[edge['to']]['label'][:20]
                    original = edge['original_time']
                    adjusted = edge['time_min']
                    increase = ((adjusted - original) / original) * 100
                    
                    self.log_result(f"    • {from_label} → {to_label}")
                    self.log_result(f"      {original:.1f} min → {adjusted:.1f} min (+{increase:.0f}%)")
            else:
                self.log_result(f"    No significant modifications")
            
            # Recommendations
            self.log_result(f"\n[RECOMMENDATIONS]")
            if severity in ['moderate', 'severe']:
                self.log_result(f"    ⛔ Incident detected - Route avoids affected areas")
            if avg_flow > 1.5:
                self.log_result(f"    ⚠️  Heavy traffic - Consider alternative time")
            if final_time > 45:
                self.log_result(f"    ⚠️  Long journey - Plan for rest stops")
            
            if severity == 'none' and avg_flow < 1:
                self.log_result(f"    ✅ Optimal conditions - Clear route ahead")
            
            self.log_result("\n" + "="*70)
            
        except Exception as e:
            self.log_result(f"\n[ERROR] {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Calculation Error", str(e))
    
    def create_path_features(self, origin: str, destination: str, severity: str, avg_flow: float) -> np.ndarray:
        """Create feature vector for LSTM (legacy - kept for compatibility)"""
        path = np.random.rand(30, 15)
        path[:, 5] = ['none', 'minor', 'moderate', 'severe'].index(severity) / 3.0
        path[:, 2] = avg_flow / 2.0
        return path
    
    def create_path_features_from_path(self, path: List[str], edges: List[Dict], severity: str) -> np.ndarray:
        """
        Create realistic LSTM features from actual path
        
        Args:
            path: List of node IDs in route
            edges: List of edge dictionaries (with modifications)
            severity: Incident severity
        
        Returns:
            numpy array of shape (30, 15) for LSTM input
        """
        features = np.zeros((30, 15))
        
        # Build edge lookup
        edge_dict = {}
        for edge in edges:
            key = (edge['from'], edge['to'])
            edge_dict[key] = edge
        
        # Fill features for each segment (up to 30)
        for i in range(min(len(path) - 1, 30)):
            from_node = path[i]
            to_node = path[i + 1]
            
            # Get edge info
            edge = edge_dict.get((from_node, to_node))
            if edge:
                features[i, 0] = edge.get('time_min', 0)  # Travel time
                features[i, 1] = edge.get('original_time', 0)  # Original time
                features[i, 2] = edge.get('flow_factor', 1.0)  # Traffic flow
                features[i, 3] = edge.get('incident_factor', 1.0)  # Incident impact
                features[i, 4] = i / 30.0  # Segment index normalized
                features[i, 5] = ['none', 'minor', 'moderate', 'severe'].index(severity) / 3.0
                
                # Add random realistic features for other columns
                features[i, 6:] = np.random.rand(9) * 0.5 + 0.5
        
        return features
    
    def find_shortest_path_dijkstra(self, origin: str, destination: str, edges: List[Dict]) -> Tuple[List[str], float]:
        """
        Dijkstra's algorithm with modified edge weights
        
        Args:
            origin: Start node ID
            destination: End node ID
            edges: List of edges with modified 'time_min' weights
        
        Returns:
            tuple: (path as list of node IDs, total time)
        """
        import heapq
        
        # Build adjacency list from modified edges
        graph = {}
        for edge in edges:
            from_node = edge['from']
            to_node = edge['to']
            weight = edge['time_min']  # ⭐ Using MODIFIED weight
            
            if from_node not in graph:
                graph[from_node] = []
            graph[from_node].append((to_node, weight))
        
        # Dijkstra's algorithm
        distances = {node: float('inf') for node in self.nodes.keys()}
        distances[origin] = 0
        previous = {node: None for node in self.nodes.keys()}
        pq = [(0, origin)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == destination:
                break
            
            if current_dist > distances[current]:
                continue
            
            for neighbor, weight in graph.get(current, []):
                distance = current_dist + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        # Reconstruct path
        if distances[destination] == float('inf'):
            return None, 0
        
        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        return path, distances[destination]
    
    def calculate_path_distance(self, path: List[str], edges: List[Dict]) -> float:
        """
        Calculate total distance of path
        
        Args:
            path: List of node IDs
            edges: List of edges
        
        Returns:
            float: Total distance in kilometers
        """
        from math import radians, cos, sin, sqrt, atan2
        
        total_distance = 0.0
        R = 6371  # Earth radius in km
        
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            
            lat1 = radians(self.nodes[from_node]['lat'])
            lon1 = radians(self.nodes[from_node]['lon'])
            lat2 = radians(self.nodes[to_node]['lat'])
            lon2 = radians(self.nodes[to_node]['lon'])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance = R * c
            
            total_distance += distance
        
        return total_distance
    
    def estimate_distance(self, origin: str, destination: str) -> float:
        """Estimate distance between two nodes"""
        from math import radians, cos, sin, sqrt, atan2
        
        lat1 = radians(self.nodes[origin]['lat'])
        lon1 = radians(self.nodes[origin]['lon'])
        lat2 = radians(self.nodes[destination]['lat'])
        lon2 = radians(self.nodes[destination]['lon'])
        
        R = 6371
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def log_result(self, message: str):
        """Add message to results text area"""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
    
    def view_map(self):
        """Open interactive map in browser"""
        map_path = Path("heritage_map_roads.html")
        if map_path.exists():
            webbrowser.open(str(map_path.absolute()))
        else:
            messagebox.showinfo("Map Not Found", 
                              "Generate map first:\npython visualize_assignment_folium_roads_knearest.py")
    
    def export_results(self):
        """Export results to JSON"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            results = {
                'timestamp': datetime.now().isoformat(),
                'origin': self.origin_node.get(),
                'destination': self.destination_node.get(),
                'severity': self.predicted_severity,
                'results': self.results_text.get(1.0, tk.END)
            }
            
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            messagebox.showinfo("Export Complete", f"Results saved to:\n{file_path}")


def main():
    """Run the GUI application"""
    root = tk.Tk()
    app = TrafficICS_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()