"""
Modern Dark-Themed GUI - Traffic Incident Classification System
"""

import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import copy

sys.path.append(str(Path(__file__).parent.parent))

from src.models.image_models import (
    load_resnet18, load_mobilenet, load_efficientnet,
    predict_severity, get_edge_multiplier, CLASS_NAMES
)
from src.graph_construction import parse_road_network
from src.pathfinding_integration import PathfindingIntegration


# ============================================================================
# MODERN UI COMPONENTS
# ============================================================================

class ModernButton(tk.Canvas):
    """Custom modern button with rounded corners and hover effects"""
    
    def __init__(self, parent, text, command, bg='#3b82f6', hover='#2563eb',
                 fg='white', width=200, height=45, radius=10):
        super().__init__(parent, width=width, height=height, 
                        bg=parent['bg'], highlightthickness=0, cursor='hand2')
        
        self.text = text
        self.command = command
        self.bg_color = bg
        self.hover_color = hover
        self.fg_color = fg
        self.radius = radius
        self.width = width
        self.height = height
        
        self.draw_button(bg)
        
        self.bind('<Enter>', lambda e: self.draw_button(hover))
        self.bind('<Leave>', lambda e: self.draw_button(bg))
        self.bind('<Button-1>', lambda e: command())
    
    def draw_button(self, color):
        """Draw rounded rectangle button"""
        self.delete('all')
        
        r = self.radius
        w, h = self.width, self.height
        
        points = [r, 0, w-r, 0, w, 0, w, r, w, h-r, w, h, w-r, h, r, h, 0, h, 0, h-r, 0, r, 0, 0]
        self.create_polygon(points, fill=color, smooth=True, outline='')
        self.create_text(w/2, h/2, text=self.text, fill=self.fg_color, font=('Segoe UI Semibold', 11))


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class TrafficICS_GUI:
    """Modern Dark-Themed Traffic ICS GUI"""
    
    COLORS = {
        'bg': '#0a0a0a',
        'card': '#1a1a1a',
        'card_hover': '#242424',
        'accent': '#2d2d2d',
        'border': '#3d3d3d',
        'primary': '#3b82f6',
        'primary_hover': '#2563eb',
        'success': '#10b981',
        'success_hover': '#059669',
        'warning': '#f59e0b',
        'warning_hover': '#d97706',
        'danger': '#ef4444',
        'text': '#f5f5f5',
        'text_dim': '#9ca3af',
        'text_darker': '#6b7280',
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic ICS - Intelligent Route Optimization")
        self.root.geometry("1600x950")
        self.root.configure(bg=self.COLORS['bg'])
        
        # State
        self.selected_image = None
        self.predictions = {}
        self.origin_var = tk.StringVar()
        self.dest_var = tk.StringVar()
        self.original_graph = None  # Store unmodified graph
        
        # Load models and graph
        self.load_models()
        self.load_graph()
        
        # Set default origin/destination
        if self.display_options:
            self.origin_var.set(self.display_options[0])
            self.dest_var.set(self.display_options[-1])
        
        # Build GUI
        self.create_widgets()
    
    def load_models(self):
        """Load 3 image models"""
        print("Loading models...")
        try:
            self.resnet = load_resnet18('models/resnet18_model.pth')
            self.mobilenet = load_mobilenet('models/mobilenet_model.pth')
            self.efficientnet = load_efficientnet('models/efficientnet_model.pth')
            print("[OK] All 3 models loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {e}")
            sys.exit(1)
    
    def load_graph(self):
        """Load road network and create node display names"""
        nodes, ways, cameras, meta = parse_road_network('heritage_assignment_15_time_asymmetric-1.txt')
        
        # Build adjacency list
        self.graph = {}
        for way in ways:
            if way['from'] not in self.graph:
                self.graph[way['from']] = []
            self.graph[way['from']].append((way['to'], way['time_min']))
        
        # STORE ORIGINAL GRAPH (important for severity comparison)
        self.original_graph = copy.deepcopy(self.graph)
        
        self.nodes = nodes
        
        # Create display names: "NodeID: Landmark Name"
        self.node_display_names = {}
        self.display_to_id = {}
        
        for node_id, info in nodes.items():
            display_name = f"{node_id}: {info['label']}"
            self.node_display_names[node_id] = display_name
            self.display_to_id[display_name] = node_id
        
        # Sorted list for dropdowns
        self.display_options = sorted(self.node_display_names.values(), 
                                     key=lambda x: int(x.split(':')[0]))
        
        self.pathfinder = PathfindingIntegration(self.graph, nodes)
        print(f"[OK] Graph loaded: {len(nodes)} nodes")
    
    def create_widgets(self):
        """Build modern dark UI"""
        
        # ===== HEADER =====
        header = tk.Frame(self.root, bg=self.COLORS['card'], height=100)
        header.pack(fill=tk.X, padx=0, pady=0)
        header.pack_propagate(False)
        
        header_content = tk.Frame(header, bg=self.COLORS['card'])
        header_content.place(relx=0.5, rely=0.5, anchor='center')
        
        tk.Label(
            header_content,
            text="🚗",
            font=('Segoe UI', 36),
            bg=self.COLORS['card'],
            fg=self.COLORS['text']
        ).pack(side=tk.LEFT, padx=(0, 15))
        
        title_frame = tk.Frame(header_content, bg=self.COLORS['card'])
        title_frame.pack(side=tk.LEFT)
        
        tk.Label(
            title_frame,
            text="Traffic Incident Classification System",
            font=('Segoe UI', 26, 'bold'),
            bg=self.COLORS['card'],
            fg=self.COLORS['text']
        ).pack(anchor='w')
        
        tk.Label(
            title_frame,
            text="AI-Powered Route Optimization • Kuching Heritage Area",
            font=('Segoe UI', 11),
            bg=self.COLORS['card'],
            fg=self.COLORS['text_dim']
        ).pack(anchor='w', pady=(2, 0))
        
        # ===== MAIN CONTAINER =====
        main = tk.Frame(self.root, bg=self.COLORS['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        # ===== TOP ROW =====
        top_row = tk.Frame(main, bg=self.COLORS['bg'])
        top_row.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.create_image_panel(top_row)
        self.create_path_panel(top_row)
        
        # ===== BOTTOM ROW =====
        self.create_results_panel(main)
    
    def create_image_panel(self, parent):
        """Modern image analysis panel"""
        panel = tk.Frame(parent, bg=self.COLORS['bg'])
        panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        title_frame = tk.Frame(panel, bg=self.COLORS['bg'])
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(title_frame, text="📸", font=('Segoe UI', 20), bg=self.COLORS['bg'], fg=self.COLORS['text']).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(title_frame, text="Image Analysis", font=('Segoe UI', 18, 'bold'), bg=self.COLORS['bg'], fg=self.COLORS['text']).pack(side=tk.LEFT)
        
        card = tk.Frame(panel, bg=self.COLORS['card'], relief=tk.FLAT)
        card.pack(fill=tk.BOTH, expand=True)
        
        card_inner = tk.Frame(card, bg=self.COLORS['card'])
        card_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        upload_btn = ModernButton(card_inner, "📁  Upload Incident Image", self.upload_image,
                                  bg=self.COLORS['primary'], hover=self.COLORS['primary_hover'], width=300, height=50, radius=12)
        upload_btn.pack(pady=(0, 15))
        
        image_container = tk.Frame(card_inner, bg=self.COLORS['accent'])
        image_container.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.image_label = tk.Label(image_container, text="No image selected\n\n📷\n\nClick upload to begin",
                                    font=('Segoe UI', 12), bg=self.COLORS['accent'], fg=self.COLORS['text_dim'], height=14)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        analyze_btn = ModernButton(card_inner, "🔍  Analyze with 3 Models", self.analyze_with_all_models,
                                   bg=self.COLORS['success'], hover=self.COLORS['success_hover'], width=300, height=50, radius=12)
        analyze_btn.pack(pady=(0, 15))
        
        results_header = tk.Frame(card_inner, bg=self.COLORS['card'])
        results_header.pack(fill=tk.X, pady=(10, 5))
        
        tk.Label(results_header, text="Predictions", font=('Segoe UI', 13, 'bold'), bg=self.COLORS['card'], fg=self.COLORS['text']).pack(side=tk.LEFT)
        
        results_container = tk.Frame(card_inner, bg=self.COLORS['accent'])
        results_container.pack(fill=tk.BOTH, expand=True)
        
        self.results_label = tk.Label(results_container, text="Awaiting analysis...", font=('Consolas', 9),
                                      bg=self.COLORS['accent'], fg=self.COLORS['text_dim'], justify=tk.LEFT, anchor='nw')
        self.results_label.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
    
    def create_path_panel(self, parent):
        """Modern pathfinding panel with landmark names"""
        panel = tk.Frame(parent, bg=self.COLORS['bg'])
        panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(15, 0))
        
        title_frame = tk.Frame(panel, bg=self.COLORS['bg'])
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(title_frame, text="🗺️", font=('Segoe UI', 20), bg=self.COLORS['bg'], fg=self.COLORS['text']).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(title_frame, text="Route Planning", font=('Segoe UI', 18, 'bold'), bg=self.COLORS['bg'], fg=self.COLORS['text']).pack(side=tk.LEFT)
        
        card = tk.Frame(panel, bg=self.COLORS['card'])
        card.pack(fill=tk.BOTH, expand=True)
        
        card_inner = tk.Frame(card, bg=self.COLORS['card'])
        card_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Origin selection
        tk.Label(card_inner, text="Origin (Start Point)", font=('Segoe UI', 11, 'bold'),
                bg=self.COLORS['card'], fg=self.COLORS['text']).pack(fill=tk.X, pady=(0, 5))
        
        origin_frame = tk.Frame(card_inner, bg=self.COLORS['accent'], height=45)
        origin_frame.pack(fill=tk.X, pady=(0, 20))
        origin_frame.pack_propagate(False)
        
        # CHANGED: Use display names ("1: Fort Margherita")
        origin_combo = ttk.Combobox(origin_frame, textvariable=self.origin_var,
                                    values=self.display_options, state='readonly', font=('Segoe UI', 10))
        origin_combo.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        
        # Destination selection
        tk.Label(card_inner, text="Destination (Goal)", font=('Segoe UI', 11, 'bold'),
                bg=self.COLORS['card'], fg=self.COLORS['text']).pack(fill=tk.X, pady=(0, 5))
        
        dest_frame = tk.Frame(card_inner, bg=self.COLORS['accent'], height=45)
        dest_frame.pack(fill=tk.X, pady=(0, 20))
        dest_frame.pack_propagate(False)
        
        # CHANGED: Use display names ("10: Sarawak Museum")
        dest_combo = ttk.Combobox(dest_frame, textvariable=self.dest_var,
                                  values=self.display_options, state='readonly', font=('Segoe UI', 10))
        dest_combo.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        
        calc_btn = ModernButton(card_inner, "🚀  Find Top-5 Routes", self.run_pathfinding,
                               bg=self.COLORS['warning'], hover=self.COLORS['warning_hover'], width=300, height=50, radius=12)
        calc_btn.pack(pady=(0, 20))
        
        info_frame = tk.Frame(card_inner, bg=self.COLORS['accent'])
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        info_inner = tk.Frame(info_frame, bg=self.COLORS['accent'])
        info_inner.pack(fill=tk.X, padx=15, pady=15)
        
        tk.Label(info_inner, text="🤖 AI Models Active", font=('Segoe UI', 11, 'bold'),
                bg=self.COLORS['accent'], fg=self.COLORS['text']).pack(anchor='w', pady=(0, 8))
        
        for info in ["• ResNet-18 - Baseline architecture", "• MobileNet-V2 - Lightweight model", "• EfficientNet-B0 - State-of-the-art"]:
            tk.Label(info_inner, text=info, font=('Segoe UI', 9), bg=self.COLORS['accent'], fg=self.COLORS['text_dim']).pack(anchor='w', pady=2)
    
    def create_results_panel(self, parent):
        """Modern results panel"""
        panel = tk.Frame(parent, bg=self.COLORS['bg'])
        panel.pack(fill=tk.BOTH, expand=True)
        
        title_frame = tk.Frame(panel, bg=self.COLORS['bg'])
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(title_frame, text="📊", font=('Segoe UI', 20), bg=self.COLORS['bg'], fg=self.COLORS['text']).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(title_frame, text="Analysis Results", font=('Segoe UI', 18, 'bold'), bg=self.COLORS['bg'], fg=self.COLORS['text']).pack(side=tk.LEFT)
        
        card = tk.Frame(panel, bg=self.COLORS['card'], height=250)
        card.pack(fill=tk.BOTH, expand=True)
        card.pack_propagate(False)
        
        text_container = tk.Frame(card, bg=self.COLORS['accent'])
        text_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        scrollbar = tk.Scrollbar(text_container, bg=self.COLORS['accent'])
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.path_results = tk.Text(text_container, font=('Consolas', 9), bg=self.COLORS['accent'], fg=self.COLORS['text'],
                                    wrap=tk.WORD, yscrollcommand=scrollbar.set, relief=tk.FLAT, bd=0, padx=15, pady=15,
                                    insertbackground=self.COLORS['text'], selectbackground=self.COLORS['primary'], selectforeground=self.COLORS['text'])
        self.path_results.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.path_results.yview)
        
        self.path_results.insert(tk.END, "Select origin/destination and click 'Find Top-5 Routes' to begin...")
    
    def upload_image(self):
        """Upload image"""
        file_path = filedialog.askopenfilename(title="Select Incident Image", filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")])
        
        if file_path:
            self.selected_image = file_path
            img = Image.open(file_path)
            img.thumbnail((450, 350))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
    
    def analyze_with_all_models(self):
        """Analyze with 3 models"""
        if not self.selected_image:
            messagebox.showwarning("No Image", "Please upload an image first")
            return
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img = Image.open(self.selected_image).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        # Predict
        sev1, conf1 = predict_severity(self.resnet, img_tensor)
        sev2, conf2 = predict_severity(self.mobilenet, img_tensor)
        sev3, conf3 = predict_severity(self.efficientnet, img_tensor)
        
        self.predictions = {
            'ResNet-18': (sev1, conf1),
            'MobileNet-V2': (sev2, conf2),
            'EfficientNet-B0': (sev3, conf3)
        }
        
        # Display
        result = "═══ 3-MODEL PREDICTIONS ═══\n\n"
        for model, (sev, conf) in self.predictions.items():
            result += f"┌─ {model}\n"
            result += f"│  Severity:   {sev.upper()}\n"
            result += f"│  Confidence: {conf*100:.1f}%\n"
            result += f"└─ Multiplier: {get_edge_multiplier(sev)}x\n\n"
        
        # Ensemble
        severities = [s for s, _ in self.predictions.values()]
        ensemble = max(set(severities), key=severities.count)
        result += f"═══ ENSEMBLE RESULT ═══\n"
        result += f"Final: {ensemble.upper()} ({get_edge_multiplier(ensemble)}x)"
        
        self.results_label.config(text=result)
    
    def run_pathfinding(self):
        """Find top-5 paths with severity impact comparison"""
        # Convert display names to node IDs
        origin_display = self.origin_var.get()
        dest_display = self.dest_var.get()
        
        if not origin_display or not dest_display:
            messagebox.showwarning("Error", "Please select origin and destination")
            return
        
        start = self.display_to_id.get(origin_display)
        goal = self.display_to_id.get(dest_display)
        
        if not start or not goal:
            messagebox.showwarning("Error", "Invalid selection")
            return
        
        if start == goal:
            messagebox.showwarning("Error", "Origin and destination must be different")
            return
        
        self.path_results.delete(1.0, tk.END)
        
        # Show selected route with landmark names
        self.path_results.insert(tk.END, f"Route: {origin_display} → {dest_display}\n")
        self.path_results.insert(tk.END, "═"*70 + "\n\n")
        
        # SEVERITY IMPACT: Compare with and without incident
        if self.predictions:
            severities = [s for s, _ in self.predictions.values()]
            ensemble = max(set(severities), key=severities.count)
            multiplier = get_edge_multiplier(ensemble)
            
            # Create modified graph (DON'T modify original)
            modified_graph = {}
            for node in self.original_graph:
                modified_graph[node] = [(n, w * multiplier) for n, w in self.original_graph[node]]
            
            # Show severity impact
            self.path_results.insert(tk.END, f"🚨 INCIDENT DETECTED: {ensemble.upper()}\n")
            self.path_results.insert(tk.END, f"   Travel Time Multiplier: {multiplier}x\n")
            self.path_results.insert(tk.END, f"   Impact: +{(multiplier-1)*100:.0f}% travel time increase\n\n")
            
            # Run pathfinding on ORIGINAL graph first
            pathfinder_original = PathfindingIntegration(self.original_graph, self.nodes)
            original_result = pathfinder_original.astar(start, goal)
            original_cost = original_result[1] if original_result[0] else float('inf')
            
            # Run pathfinding on MODIFIED graph
            pathfinder_modified = PathfindingIntegration(modified_graph, self.nodes)
            all_results = pathfinder_modified.find_top3_algorithms(start, goal)
            
            # Show comparison
            self.path_results.insert(tk.END, f"⚙️  BEFORE/AFTER COMPARISON:\n")
            self.path_results.insert(tk.END, f"   Without incident: {original_cost:.2f} min\n")
            if all_results:
                best_cost = all_results[0]['cost']
                delay = best_cost - original_cost
                self.path_results.insert(tk.END, f"   With {ensemble} incident: {best_cost:.2f} min\n")
                self.path_results.insert(tk.END, f"   Additional delay: +{delay:.2f} min ({delay/original_cost*100:.0f}%)\n\n")
        else:
            # No incident detected
            self.path_results.insert(tk.END, "ℹ️  No incident detected (using normal traffic conditions)\n\n")
            pathfinder_original = PathfindingIntegration(self.original_graph, self.nodes)
            all_results = pathfinder_original.find_top3_algorithms(start, goal)
        
        # Display top-5 routes
        top5 = all_results[:5] if len(all_results) >= 5 else all_results
        
        self.path_results.insert(tk.END, "═"*70 + "\n")
        self.path_results.insert(tk.END, "TOP-5 OPTIMAL ROUTES\n")
        self.path_results.insert(tk.END, "═"*70 + "\n\n")
        
        for i, result in enumerate(top5, 1):
            self.path_results.insert(tk.END, f"┌─ Route {i}: {result['algorithm']}\n")
            self.path_results.insert(tk.END, f"│  Travel Time:    {result['cost']:.2f} min\n")
            self.path_results.insert(tk.END, f"│  Nodes Expanded: {result['nodes_expanded']}\n")
            self.path_results.insert(tk.END, f"│  Path Length:    {result['path_length']} segments\n")
            
            # Show path with landmark names (first 4 nodes)
            path_preview = []
            for node_id in result['path'][:4]:
                path_preview.append(self.node_display_names.get(node_id, node_id))
            self.path_results.insert(tk.END, f"└─ Path: {' → '.join(path_preview)}...\n\n")


def main():
    root = tk.Tk()
    
    # Dark theme for comboboxes
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TCombobox', fieldbackground='#2d2d2d', background='#3d3d3d',
                   foreground='#f5f5f5', arrowcolor='#9ca3af', borderwidth=0, relief='flat')
    style.map('TCombobox', fieldbackground=[('readonly', '#2d2d2d')],
             selectbackground=[('readonly', '#3b82f6')], selectforeground=[('readonly', '#f5f5f5')])
    
    app = TrafficICS_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()