"""Draw the two architectures as SVG (light card, dark text)."""
from common import MEDIA

W, H = 900, 420
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Approach A uses two separate models; Approach B uses one shared encoder with two heads">',
         '<style>text{font:13px sans-serif;fill:#24292F}.t{font-weight:bold;font-size:15px}.s{font-size:11px;fill:#57606A}</style>',
         f'<rect width="{W}" height="{H}" rx="10" fill="#F6F8FA"/>',
         '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8z" fill="#57606A"/></marker></defs>']

def box(x, y, w, h, label, fill, sub=None):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="#57606A"/>')
    parts.append(f'<text x="{x+w/2}" y="{y+h/2+(0 if sub else 5)}" text-anchor="middle">{label}</text>')
    if sub:
        parts.append(f'<text class="s" x="{x+w/2}" y="{y+h/2+16}" text-anchor="middle">{sub}</text>')

def arrow(x1, y1, x2, y2):
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#57606A" stroke-width="1.5" marker-end="url(#a)"/>')

# Approach A
parts.append('<text class="t" x="20" y="28">Approach A: two separate models</text>')
box(20, 140, 90, 40, "Tweet", "#FFFFFF")
box(170, 70, 170, 50, "FinBERT", "#CCE4F4", "fine-tuned, 3 classes")
box(170, 200, 170, 50, "BERTweet", "#FBE7B5", "fine-tuned, 21 classes")
box(400, 70, 120, 50, "Sentiment", "#FFFFFF")
box(400, 200, 120, 50, "Company", "#FFFFFF")
arrow(110, 160, 170, 95); arrow(110, 160, 170, 225); arrow(340, 95, 400, 95); arrow(340, 225, 400, 225)
parts.append('<text class="s" x="170" y="275">a helper function returns both outputs for one tweet</text>')
# Approach B
parts.append('<text class="t" x="560" y="28">Approach B: shared encoder, two heads</text>')
box(560, 140, 80, 40, "Tweet", "#FFFFFF")
box(690, 120, 190, 80, "FinTwitBERT encoder", "#CCE4F4", "shared, fine-tuned")
box(690, 40, 190, 45, "Sentiment head", "#D4EFE5", "pretrained, fine-tuned")
box(690, 235, 190, 45, "Company head", "#FBE7B5", "new, trained from scratch")
arrow(640, 160, 690, 160); arrow(785, 120, 785, 85); arrow(785, 200, 785, 235)
parts.append('<text class="s" x="560" y="320">Two copies (L, R) are trained; a consistency loss ties their</text>')
parts.append('<text class="s" x="560" y="336">predictions on tweets that lack the label for a task.</text>')
parts.append('<text class="s" x="20" y="400">Diagram drawn from analysis/fig_architecture.py.</text>')
parts.append('</svg>')
(MEDIA / "architecture.svg").write_text("\n".join(parts))
