# Graphical Abstract Design Specification

## Applied Soft Computing Submission Requirements
- **Text limit**: 25 words maximum
- **Figure size**: 5×5 cm (approximately 600×600 pixels at 300 DPI)
- **Purpose**: Visually convey the research essence at a glance
- **Audience**: Journal readers browsing abstracts

---

## 25-Word Text (Version 1)

**"Deep reinforcement learning optimizes vertical UAM queueing: A2C achieves 50%+ improvement over heuristics; inverted pyramid structure outperforms normal by 9.5%; capacity paradox identified."**

**Word count**: 25 words ✓

---

## Alternative 25-Word Versions

**Version 2 (More accessible):**
"Fifteen DRL algorithms tested for urban air mobility. A2C best: 50% better than heuristics. Inverted pyramid optimal. Surprising finding: low capacity outperforms high under extreme load."

**Word count**: 27 words ✗ (needs trimming)

**Version 2 (Revised):**
"Fifteen DRL algorithms tested for urban air mobility. A2C achieves 50% improvement. Inverted pyramid optimal. Capacity paradox: low capacity outperforms high under extreme load."

**Word count**: 25 words ✓

**Version 3 (Action-oriented):**
"Deep reinforcement learning solves vertical airspace congestion. A2C algorithm delivers 50%+ gains. Inverted pyramid structure recommended. Counter-intuitive capacity paradox discovered at extreme loads."

**Word count**: 24 words ✓

---

## Visual Design Concept

### Layout Structure (5×5 cm figure)

```
┌─────────────────────────────────────────┐
│  [Problem]                              │
│  🚁 UAM Vertical Airspace              │
│  ↓                                      │
│  [Method]                               │
│  🤖 15 DRL Algorithms                   │
│  ↓                                      │
│  [Key Findings - 3 panels]             │
│  ┌──────┬──────┬──────┐               │
│  │ DRL  │Invert│Capac │               │
│  │ 50%+ │ +9.5%│Paradx│               │
│  └──────┴──────┴──────┘               │
└─────────────────────────────────────────┘
```

### Detailed Visual Elements

#### Panel 1: Problem Visualization (Top, 20% height)
- **Visual**: Stylized vertical layers (5 stacked rectangles) with drone icons
- **Color**: Sky blue gradient (light at top, darker at bottom)
- **Labels**: "L0" to "L4" on layers
- **Icon**: Small drone silhouettes showing congestion

#### Panel 2: Method Visualization (Middle, 20% height)
- **Visual**: Neural network icon + algorithm names
- **Color**: Orange/red gradient (representing DRL/AI)
- **Text**: "15 DRL Algorithms" with A2C, PPO, TD7 highlighted
- **Icon**: Brain/network symbol

#### Panel 3: Key Findings (Bottom, 60% height - 3 sub-panels)

**Sub-panel 3A: DRL Superiority (Left, 33%)**
- **Visual**: Bar chart comparison
  - Blue bar (DRL): Tall, labeled "4438"
  - Gray bar (Heuristic): Short, labeled "1876"
- **Annotation**: "+50%" with upward arrow
- **Title**: "DRL vs Heuristics"

**Sub-panel 3B: Structural Optimality (Center, 33%)**
- **Visual**: Two pyramid diagrams side-by-side
  - Left: Inverted pyramid [8,6,4,3,2] in green ✓
  - Right: Normal pyramid [2,3,4,6,8] in red ✗
- **Annotation**: "+9.5%" above inverted
- **Title**: "Inverted Optimal"

**Sub-panel 3C: Capacity Paradox (Right, 33%)**
- **Visual**: Line graph showing inverted U-curve
  - X-axis: K=10, 20, 30, 40
  - Y-axis: Reward
  - Peak at K=10, collapse at K=30+
- **Annotation**: "Less is More!" with surprise icon
- **Title**: "Capacity Paradox"

---

## Color Scheme

### Primary Colors
- **Sky Blue** (#87CEEB): UAM/airspace theme
- **Neural Orange** (#FF6B35): DRL/AI theme
- **Success Green** (#4CAF50): Positive results
- **Warning Red** (#F44336): Negative results/failures
- **Neutral Gray** (#9E9E9E): Baselines/comparisons

### Accent Colors
- **Gold** (#FFD700): Highlighting best results (A2C)
- **White** (#FFFFFF): Background, clean look
- **Dark Gray** (#424242): Text, labels

---

## Typography

### Font Recommendations
- **Title/Headers**: Arial Bold, 10-12pt
- **Body Text**: Arial Regular, 8-10pt
- **Annotations**: Arial Bold, 7-9pt
- **Numbers**: Arial Bold, 9-11pt

### Text Hierarchy
1. **Main title** (25 words): 10pt, centered, top
2. **Panel titles**: 9pt, bold, centered above each panel
3. **Annotations**: 8pt, bold, positioned near relevant elements
4. **Axis labels**: 7pt, regular

---

## Implementation Notes

### Software Recommendations
- **Vector graphics**: Adobe Illustrator, Inkscape (free)
- **Python plotting**: Matplotlib, Seaborn with high DPI export
- **Diagram tools**: draw.io, Figma

### Export Settings
- **Format**: PNG or TIFF (for print quality)
- **Resolution**: 300 DPI minimum
- **Size**: 1772×1772 pixels (5cm × 5cm at 300 DPI)
- **Color mode**: RGB (for digital) or CMYK (for print)

### Design Principles
1. **Simplicity**: Avoid clutter, focus on 3 key messages
2. **Visual hierarchy**: Problem → Method → Results flow
3. **Color coding**: Consistent use of colors for meaning
4. **Readability**: Large enough text/icons for 5×5cm size
5. **Self-explanatory**: Should make sense without reading paper

---

## Alternative Design: Horizontal Flow

```
┌─────────────────────────────────────────────────────────┐
│ Problem → Method → Results                              │
│ [UAM]  →  [DRL]  → [+50%] [+9.5%] [Paradox]          │
│  🚁    →   🤖   →   📊     🔺      📉                │
└─────────────────────────────────────────────────────────┘
```

This alternative uses left-to-right flow instead of top-to-bottom, which may be more intuitive for some readers.

---

## Recommended Final Version

**Text**: Version 3 (Action-oriented, 24 words)
"Deep reinforcement learning solves vertical airspace congestion. A2C algorithm delivers 50%+ gains. Inverted pyramid structure recommended. Counter-intuitive capacity paradox discovered at extreme loads."

**Visual**: Vertical layout with 3-panel findings section
- Clear visual hierarchy
- Color-coded for quick comprehension
- Balances technical detail with accessibility

---

**Next Steps for Implementation:**
1. Create vector graphics for vertical layers and pyramids
2. Generate bar chart and line graph using Matplotlib
3. Combine elements in Illustrator/Inkscape
4. Export at 300 DPI, 5×5cm size
5. Review for readability at actual size
6. Iterate based on visual balance

**Estimated time**: 2-3 hours for professional-quality graphical abstract
