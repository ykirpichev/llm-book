"""Original vector teaching figures, authored in book points (430 pt wide).

No raster assets or external artwork. Labels are measured before drawing; every
figure has an explicit canvas, readable type, and a stated scope or assumption.
Coordinates use ReportLab's bottom-left origin. Arrows indicate data/dependency
flow, except in explicitly labelled timelines and state-transition diagrams.
"""
from __future__ import annotations

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.pdfbase.pdfmetrics import stringWidth


class Figure:
    def __init__(self, height, title, fonts, palette):
        self.d = Drawing(430, height)
        self.font, self.bold = fonts
        self.p = palette
        self.nodes = []
        self.text_boxes = []
        self.segments = []
        self.rect(0, 0, 430, height, "PAPER", "PANEL", radius=8)
        self.text(18, height - 24, title, size=10, bold=True, width=394)
        self.line(18, height - 35, 412, height - 35, "TEAL")

    def rect(self, x, y, w, h, fill="WHITE", stroke="TEAL", radius=0):
        self.d.add(Rect(x, y, w, h, rx=radius, ry=radius,
                        fillColor=self.p[fill], strokeColor=self.p[stroke], strokeWidth=.7))

    def text(self, x, y, text, size=8.5, color="INK", bold=False,
             anchor="start", width=394):
        font = self.bold if bold else self.font
        measured = stringWidth(text, font, size)
        if measured > width + .1:
            raise ValueError(f"Figure label exceeds {width} pt: {text!r} ({measured:.1f})")
        left = x - measured / 2 if anchor == "middle" else x - measured if anchor == "end" else x
        self.text_boxes.append((left, y - size * .25, left + measured, y + size, text))
        self.d.add(String(x, y, text, fontName=font, fontSize=size,
                          fillColor=self.p[color], textAnchor=anchor))

    def node(self, x, y, w, h, label, tone="TEAL", fill=None, size=8.8):
        self.nodes.append((x, y, x + w, y + h, label))
        self.rect(x, y, w, h, fill or "PALE_" + tone, tone, radius=5)
        lines = label.split("\n")
        leading = 12
        if len(lines) * leading > h - 10:
            raise ValueError(f"Figure label too tall: {label!r}")
        for i, line in enumerate(lines):
            self.text(x + w / 2, y + h / 2 + (len(lines)-1)*leading/2 - 3 - i*leading,
                      line, size=size, bold=i == 0, anchor="middle", width=w-12)

    def line(self, x1, y1, x2, y2, tone="TEAL", dashed=False):
        self.segments.append((x1, y1, x2, y2))
        self.d.add(Line(x1, y1, x2, y2, strokeColor=self.p[tone],
                        strokeWidth=1.2, strokeDashArray=[3, 3] if dashed else None))

    def arrow(self, points, tone="TEAL"):
        import math
        for a, b in zip(points, points[1:]):
            self.line(*a, *b, tone)
        (x1, y1), (x2, y2) = points[-2:]
        angle = math.atan2(y2-y1, x2-x1)
        coords = [x2, y2]
        for offset in [-.5, .5]:
            coords.extend([x2-5*math.cos(angle+offset), y2-5*math.sin(angle+offset)])
        self.d.add(Polygon(coords, fillColor=self.p[tone], strokeColor=None))

    def footer(self, first, second=None):
        self.text(18, 27 if second else 17, first, size=8.2, color="MUTED")
        if second:
            self.text(18, 15, second, size=8.2, color="MUTED")

    def finish(self, width):
        for x0, y0, x1, y1, label in self.text_boxes:
            if x0 < 0 or y0 < 0 or x1 > 430 or y1 > self.d.height:
                raise ValueError(f"Text outside figure: {label}")
        for i, a in enumerate(self.text_boxes):
            for b in self.text_boxes[i+1:]:
                if min(a[2], b[2]) > max(a[0], b[0]) + .1 and min(a[3], b[3]) > max(a[1], b[1]) + .1:
                    raise ValueError(f"Overlapping labels: {a[4]!r}, {b[4]!r}")
        for x1, y1, x2, y2 in self.segments:
            for left, bottom, right, top, label in self.text_boxes:
                if x1 == x2 and left < x1 < right and min(y1, y2) < top and max(y1, y2) > bottom:
                    raise ValueError(f"Vertical connector crosses label: {label!r}")
                if y1 == y2 and bottom < y1 < top and min(x1, x2) < right and max(x1, x2) > left:
                    raise ValueError(f"Horizontal connector crosses label: {label!r}")
        # Detect accidental node overlap before the PDF reaches visual review.
        for i, a in enumerate(self.nodes):
            for b in self.nodes[i+1:]:
                if min(a[2], b[2]) > max(a[0], b[0]) + .1 and min(a[3], b[3]) > max(a[1], b[1]) + .1:
                    raise ValueError(f"Overlapping nodes: {a[4]!r}, {b[4]!r}")
        scale = min(1, width / 430)
        self.d.scale(scale, scale)
        self.d.width *= scale
        self.d.height *= scale
        return self.d


def token_alignment(f):
    xs = [120, 212, 304]
    f.text(18, 181, "Input", bold=True)
    f.text(18, 121, "Prediction", bold=True)
    f.text(18, 61, "Target", bold=True)
    for i, (inp, target) in enumerate(zip(["BOS", "A", "B"], ["A", "B", "EOS"])):
        x = xs[i]
        f.node(x, 164, 76, 30, inp)
        f.node(x, 104, 76, 30, "p(next)", tone="GOLD")
        f.node(x, 44, 76, 30, target, tone="CORAL")
        f.arrow([(x+38, 164), (x+38, 134)])
        f.arrow([(x+38, 104), (x+38, 74)], "CORAL")
    f.footer("Targets shift by one position; the causal mask hides future inputs.")


def decoder_block(f):
    # Two residual sublayers. The bypass terminates at the addition, not the norm.
    for y, label, inp, out in [(172, "Attention", "X", "U"), (73, "MLP", "U", "X next")]:
        f.node(18, y, 48, 32, inp, fill="WHITE")
        f.node(100, y, 70, 32, "Norm")
        f.node(200, y, 103, 32, label, tone="GOLD")
        f.node(339, y, 72, 32, "Add: " + out, tone="CORAL")
        for a,b in [(66,100),(170,200),(303,339)]:
            f.arrow([(a,y+16),(b,y+16)])
        f.arrow([(42,y+32),(42,y+55),(375,y+55),(375,y+32)], "CORAL")
        f.text(210, y+61, "residual bypass", anchor="middle", color="CORAL")
    f.footer("Pre-norm block: attention mixes positions; the MLP mixes channels.",
             "The output U of the first sublayer is the input of the second.")


def attention_state_map(f):
    rows = [(219,"Dense / GQA", "read every allowed token", [1]*8),
            (164,"Local window", "read recent tokens", [0,0,0,0,1,1,1,1]),
            (109,"Sparse selection", "read selected blocks", [1,0,0,1,0,0,1,1])]
    for y, label, note, active in rows:
        f.text(18,y+12,label,bold=True,width=118)
        for j, used in enumerate(active):
            f.rect(150+j*24,y,20,27,"TEAL" if used else "WHITE", "TEAL")
        f.text(150,y-13,note,color="MUTED",size=8.2)
    f.node(18,42,176,37,"MLA: smaller state per token",tone="GOLD",size=8.4)
    f.node(214,42,198,37,"Recurrent: fixed-size matrix state",tone="CORAL",size=8.4)
    f.footer("Filled cells are read positions, not a promise of reduced storage.",
             "Head sharing, token selection, compression, and recurrence differ.")


def optimization_loop(f):
    f.node(18,147,106,43,"Batch + masks\ntraining examples")
    f.node(160,147,106,43,"Forward + loss\ncurrent weights",tone="GOLD")
    f.node(302,147,110,43,"Backward\ngradients",tone="CORAL")
    f.node(160,58,180,43,"Optimizer + schedule\nweights, moments, step")
    f.arrow([(124,168),(160,168)])
    f.arrow([(266,168),(302,168)])
    f.arrow([(357,147),(357,79),(340,79)],"CORAL")
    f.arrow([(213,101),(213,147)])
    f.text(223,123,"new weights",size=8.2,width=110)
    f.footer("Learning rate, clipping, precision, and batch statistics affect the loop.")


def dedup_clusters(f):
    f.node(18,154,116,44,"Fingerprints\nfind candidates")
    f.node(160,154,110,44,"Verify edges\nsimilar pairs",tone="GOLD")
    f.node(302,154,110,44,"Components\ngroup records",tone="CORAL")
    f.arrow([(134,176),(160,176)])
    f.arrow([(270,176),(302,176)])
    for x,lab in [(70,"A"),(180,"B"),(290,"C")]:
        f.node(x,74,48,32,lab,fill="WHITE")
    f.line(118,90,180,90)
    f.line(228,90,290,90)
    f.text(139,103,"similar",anchor="middle",size=8.2)
    f.text(249,103,"similar",anchor="middle",size=8.2)
    f.footer("A and C can be in one component without being similar to each other.",
             "Choose representatives and split policies at the intended group level.")


def synthetic_gate(f):
    f.node(18,141,104,45,"Seeds\ncoverage plan")
    f.node(160,141,110,45,"Generator\ncandidate traces")
    f.node(308,141,104,45,"Verifier\ncheck outcomes",tone="GOLD")
    f.node(232,52,180,44,"Deduplicate + balance\nversioned training mixture")
    f.node(18,52,168,44,"Rejected / uncertain\ninspect failure modes",tone="CORAL")
    f.arrow([(122,163),(160,163)])
    f.arrow([(270,163),(308,163)])
    f.arrow([(372,141),(372,96)])
    f.text(380,115,"pass",size=8.2,width=32)
    f.arrow([(330,141),(330,117),(102,117),(102,96)],"CORAL")
    f.footer("Audit accepted examples against untouched evaluation data before release.")


def lora_paths(f):
    f.node(18,108,52,38,"x",fill="WHITE")
    f.node(178,158,113,39,"Frozen W",tone="GOLD")
    f.node(115,64,89,39,"Low-rank A")
    f.node(238,64,89,39,"Low-rank B")
    f.node(360,108,52,38,"+",tone="CORAL")
    f.arrow([(70,127),(88,127),(88,177),(178,177)],"GOLD")
    f.arrow([(291,177),(386,177),(386,146)],"GOLD")
    f.arrow([(70,127),(88,127),(88,83),(115,83)])
    f.arrow([(204,83),(238,83)])
    f.arrow([(327,83),(386,83),(386,108)])
    f.text(219,119,"rank r",anchor="middle",size=8.2)
    f.footer("Row-vector convention: y = xW + s(xA)B; s is the adapter scale.",
             "Only A and B receive optimizer updates in ordinary LoRA.")


def policy_learning(f):
    f.node(18,159,110,47,"Behavior policy\nsample trajectories")
    f.node(160,159,110,47,"Reward / verifier\nscore outcomes",tone="GOLD",size=8.4)
    f.node(302,159,110,47,"Advantages\nrelative signal",tone="CORAL")
    f.node(160,62,110,47,"Learner\nupdate policy")
    f.node(18,62,110,47,"Reference policy\nregularization",tone="GOLD",size=8.4)
    f.arrow([(128,182),(160,182)])
    f.arrow([(270,182),(302,182)])
    f.arrow([(357,159),(357,85),(270,85)],"CORAL")
    f.arrow([(128,85),(160,85)],"GOLD")
    f.arrow([(74,159),(74,132),(185,132),(185,109)])
    f.text(83,141,"samples + old log p",size=8.2,width=182)
    f.text(285,122,"advantages",size=8.2,width=126)
    f.arrow([(215,62),(215,46),(8,46),(8,218),(74,218),(74,206)])
    f.text(18,51,"publish weights",size=8.2,width=182)
    f.footer("The reference is a regularization anchor; the behavior policy made the data.",
             "PPO and GRPO differ in how they estimate and normalize advantages.")


def kv_page_sharing(f):
    f.node(18,186,104,40,"Request A\nlogical blocks")
    f.node(18,108,104,40,"Request B\nlogical blocks",tone="CORAL")
    f.node(222,139,190,42,"P0 | P1\nshared immutable prefix")
    f.node(222,202,190,36,"P4: A's writable tail")
    f.node(222,60,190,36,"P7: B's writable tail",tone="CORAL")
    f.arrow([(122,209),(178,209),(178,220),(222,220)])
    f.arrow([(122,139),(180,139),(180,150),(222,150)],"CORAL")
    f.arrow([(122,197),(157,197),(157,170),(222,170)])
    f.arrow([(122,119),(147,119),(147,78),(222,78)],"CORAL")
    f.text(18,60,"Block tables map logical order",size=8.2,width=194)
    f.text(18,48,"onto noncontiguous physical pages.",size=8.2,width=194)
    f.footer("Shared full blocks are read-only; a shared partial block needs copy-on-write.")


def kv_handoff(f):
    labels=["Reserve", "Transfer", "Validate", "Publish"]
    for i, label in enumerate(labels):
        f.node(18+i*101,143,91,39,label,tone="GOLD" if i==2 else "TEAL")
        if i<3:f.arrow([(109+i*101,163),(119+i*101,163)])
    f.node(18,56,180,42,"Source retains recoverable KV\nuntil ownership is acknowledged",size=8.2)
    f.node(232,56,180,42,"Destination can decode\nafter valid ownership publication",tone="CORAL",size=8.2)
    f.arrow([(367,143),(367,98)],"CORAL")
    f.line(18,122,299,122,"GOLD",dashed=True)
    f.text(18,109,"failure before publish: retry or release reservation",size=8.2)
    f.footer("Transport completion alone does not grant execution ownership.")


def quantization_path(f):
    f.node(18,142,104,44,"Weights\nfull precision")
    f.node(159,142,112,44,"Group + round\nscale / zero point",tone="GOLD")
    f.node(308,142,104,44,"Packed values\n+ metadata",tone="CORAL")
    f.arrow([(122,164),(159,164)])
    f.arrow([(271,164),(308,164)],"GOLD")
    f.node(159,55,253,43,"Compatible kernel\nload, unpack or directly consume, accumulate")
    f.node(18,55,104,43,"Activations\nchosen dtype",fill="WHITE")
    f.arrow([(360,142),(360,98)],"CORAL")
    f.arrow([(122,76),(159,76)])
    f.footer("Packed bytes, scale traffic, conversion, and arithmetic all enter the budget.")


def double_buffer(f):
    # 3 us transfers, 5 us consumers; reusing a slot waits for its last read.
    x0, scale = 79, 13
    for y,label in [(168,"Load"),(114,"Use")]:
        f.text(18,y+12,label,bold=True,width=54)
    for tile,start in enumerate([0,3,8,13]):
        tone="TEAL" if tile%2==0 else "CORAL"
        f.rect(x0+start*scale,164,3*scale,28,"PALE_"+tone,tone)
        f.text(x0+(start+1.5)*scale,174,str(tile),anchor="middle",bold=True)
    for tile,start in enumerate([3,8,13,18]):
        tone="TEAL" if tile%2==0 else "CORAL"
        f.rect(x0+start*scale,110,5*scale,28,"PALE_"+tone,tone)
        f.text(x0+(start+2.5)*scale,120,str(tile),anchor="middle",bold=True)
        load_end = [3, 6, 11, 16][tile]
        f.arrow([(x0+load_end*scale,164),(x0+load_end*scale,151),
                 (x0+start*scale,151),(x0+start*scale,138)],tone)
    for t in [0,3,8,13,18,23]:
        f.text(x0+t*scale,88,str(t),anchor="middle",size=8.2)
    f.text(391,88,"us",size=8.2,width=21)
    f.text(18,61,"Sage: slot 0 (tiles 0, 2)     Terracotta: slot 1 (tiles 1, 3)",size=8.2)
    f.footer("Illustrative independent engines: four tiles finish in 23 us, versus 32 serial.",
             "A buffer is reused only after its prior consumer has finished reading it.")


def zero_shards(f):
    f.text(18,226,"Resident model state on each rank",bold=True)
    for j,title in enumerate(["Parameters","Gradients","Optimizer"]):
        f.text(172+j*94,205,title,anchor="middle",bold=True,size=8.4,width=92)
    for row,lab in enumerate(["DP","ZeRO-1","ZeRO-2","ZeRO-3"]):
        y=159-row*33
        f.text(18,y+10,lab,bold=True)
        for col in range(3):
            sharded=col>=3-row
            for part in range(4):
                f.rect(130+col*94+part*21,y,19,23,
                       "TEAL" if not sharded or part==0 else "WHITE","TEAL")
    f.footer("Four ranks illustrated. Filled: retained locally. Outline: held by peers.",
             "Stage 3 still gathers parameters for compute; activations are extra.")


def phase_sharding(f):
    # Separate diagrams: query rows during PCP, history columns during DCP.
    f.text(18,269,"PREFILL: SPLIT PROMPT QUERIES",bold=True,size=9)
    for y,lab in [(216,"Rank 0: Q rows 0-3"),(165,"Rank 1: Q rows 4-7")]:
        f.node(18,y,144,33,lab,size=8.4)
        f.node(226,y,186,33,"Causally eligible K/V",tone="GOLD",size=8.4)
        f.arrow([(226,y+16),(162,y+16)],"GOLD")
    f.text(18,147,"K/V are gathered or circulated; arrows show required reads.",size=8.2)
    f.text(18,122,"DECODE: SPLIT KV HISTORY",bold=True,size=9)
    f.node(18,60,110,42,"Same new query\nKV tokens 0-3")
    f.node(162,60,110,42,"Same new query\nKV tokens 4-7",tone="CORAL")
    f.node(306,60,106,42,"Merge partial\nattention states",tone="GOLD",size=8.4)
    f.arrow([(128,81),(145,81),(145,44),(287,44),(287,72),(306,72)])
    f.arrow([(272,91),(306,91)],"CORAL")
    f.footer("DCP sketch: one KV head (or latent cache) distributed over two ranks.")


def moe_dispatch(f):
    f.node(18,155,110,44,"Token owner\nrouter + packing")
    f.node(160,204,110,38,"Expert A\nselected MLP")
    f.node(160,137,110,38,"Expert B\nselected MLP",tone="CORAL")
    f.node(160,70,110,38,"Expert C\nnot selected",tone="GOLD")
    f.node(302,155,110,44,"Token owner\nweighted combine")
    f.arrow([(128,185),(143,185),(143,223),(160,223)])
    f.arrow([(128,169),(143,169),(143,156),(160,156)],"CORAL")
    f.arrow([(270,223),(287,223),(287,185),(302,185)])
    f.arrow([(270,156),(287,156),(287,169),(302,169)],"CORAL")
    f.text(18,118,"Dispatch",bold=True,size=8.5,width=112)
    f.text(302,118,"Return + combine",bold=True,size=8.5,width=110)
    f.footer("Top-2 example for one token. Router weights combine the returned outputs.",
             "All expert weights remain resident somewhere, including inactive experts.")


def checkpoint_commit(f):
    f.node(18,180,104,41,"Rank 0\nimmutable shard")
    f.node(18,112,104,41,"Rank 1\nimmutable shard",tone="CORAL")
    f.node(158,142,112,52,"Validate\ncoverage + hashes",tone="GOLD",size=8.4)
    f.node(304,142,108,52,"Manifest\ncomplete step")
    f.arrow([(122,200),(142,200),(142,181),(158,181)])
    f.arrow([(122,132),(142,132),(142,156),(158,156)],"CORAL")
    f.arrow([(270,168),(304,168)])
    f.node(158,50,112,40,"Atomic pointer\nlatest")
    f.node(304,50,108,40,"Restore\nverify manifest")
    f.arrow([(358,142),(358,113),(214,113),(214,90)])
    f.arrow([(270,70),(304,70)])
    f.footer("Only a durable, complete manifest becomes visible as the latest checkpoint.")


def event_time(f):
    f.text(18,187,"Arrival order",bold=True)
    for i,t in enumerate([12,17,14,23]):
        x=135+i*68
        f.node(x,169,52,31,str(t),tone="CORAL" if t==14 else "TEAL")
        if i<3:f.arrow([(x+52,184),(x+68,184)])
    f.text(18,147,"Numbers are event timestamps; arrival 3 is out of order.",size=8.2)
    f.line(40,84,390,84)
    for t in [10,14,18,20,23]:
        x=40+(t-10)*25
        f.line(x,79,x,89)
        f.text(x,65,str(t),anchor="middle",size=8.2)
    f.rect(40,90,250,23,"PALE_TEAL","TEAL")
    f.text(165,98,"window [10, 20)",anchor="middle",size=8.2)
    f.arrow([(240,133),(240,90)],"CORAL")
    f.text(250,120,"watermark 18",size=8.2,color="CORAL",width=150)
    f.footer("A watermark is an event-time progress policy, not the latest arrival.",
             "Finalize at the configured window/lateness boundary; handle late data.")


def count_min(f):
    f.text(18,222,"Query key x",bold=True)
    values=[7,5,8]
    for r,(hit,val) in enumerate(zip([1,3,2],values)):
        y=165-r*42
        f.text(18,y+8,f"hash {r+1}",bold=True)
        for col in range(5):
            f.rect(109+col*33,y,29,27,"PALE_GOLD" if col==hit else "WHITE","GOLD" if col==hit else "TEAL")
            if col==hit:f.text(123.5+col*33,y+9,str(val),anchor="middle",bold=True)
        f.arrow([(74,y+13),(109,y+13)])
    f.node(302,106,110,51,"Estimate\nmin(7, 5, 8) = 5",tone="GOLD",size=8.4)
    f.footer("One hashed counter per row is read; an update increments one per row.",
             "With nonnegative frequencies, collisions can only raise this estimate.")


def rag_pipeline(f):
    f.text(18,243,"OFFLINE: BUILD A VERSIONED INDEX",bold=True,size=9)
    for x,lab in [(18,"Documents\nrights + version"),(163,"Chunk + embed\nretain source IDs"),(308,"Index\nvector + lexical")]:
        f.node(x,179,104,45,lab,size=8.4)
    f.arrow([(122,201),(163,201)])
    f.arrow([(267,201),(308,201)])
    f.text(18,149,"ONLINE: RETRIEVE EVIDENCE FOR A QUERY",bold=True,size=9)
    for x,lab,tone in [(18,"Query","TEAL"),(121,"Retrieve +\nfilter","TEAL"),(224,"Rerank +\npack","GOLD"),(327,"Generate +\ncite","CORAL")]:
        f.node(x,80,85,43,lab,tone=tone,size=8.4)
    for a,b in [(103,121),(206,224),(309,327)]:f.arrow([(a,101),(b,101)])
    f.arrow([(360,179),(360,163),(421,163),(421,58),(164,58),(164,80)],"GOLD")
    f.footer("Source IDs travel into citations; retrieval and answer quality need separate tests.")


def multimodal_path(f):
    for y,left,right in [(198,"Image / frames","Vision encoder"),(132,"Audio features","Audio encoder"),(66,"Text","Tokenizer + embed")]:
        f.node(18,y,110,40,left,size=8.4)
        f.node(160,y,126,40,right,size=8.4,tone="GOLD")
        f.arrow([(128,y+20),(160,y+20)])
    f.node(326,115,86,75,"Align / project\n+ positions\nLM context",tone="CORAL",size=8.3)
    f.arrow([(286,218),(307,218),(307,177),(326,177)])
    f.arrow([(286,152),(326,152)])
    f.arrow([(286,86),(307,86),(307,128),(326,128)])
    f.footer("Illustrative encoder-to-language-model path; architectures may fuse differently.",
             "Keep crop transforms, timestamps, and modality-specific token budgets.")


def diffusion_blocks(f):
    f.text(18,231,"Fixed causal prefix",bold=True)
    f.text(212,231,"Current mutable block",bold=True)
    rounds=[["?","?","?","?"],["A","?","?","D"],["A","B","C","D"]]
    for row, tokens in enumerate(rounds):
        y=174-row*57
        for j in range(3):
            f.node(18+j*55,y,46,30,f"P{j}",size=8.4)
        for j,tok in enumerate(tokens):
            f.node(212+j*51,y,45,30,tok,tone="GOLD" if tok=="?" else "CORAL",size=8.4)
        if row<2:f.arrow([(401,y),(401,y-27)],"CORAL")
    f.footer("Illustrative reveal schedule. Mutable positions can attend within their block.",
             "A changed token can invalidate other cached states in that mutable block.")


def data_release(f):
    f.node(18,193,110,44,"Immutable inputs\nsource revision + ID")
    f.node(160,193,110,44,"Transforms\nversion + config")
    f.node(302,193,110,44,"Features + scores\nversioned outputs",tone="GOLD",size=8.4)
    f.arrow([(128,215),(160,215)])
    f.arrow([(270,215),(302,215)])
    f.node(18,76,144,49,"Lineage + policy catalog\nparents, rights, decisions",tone="GOLD",size=8.4)
    f.node(200,76,88,49,"Select + mix\nrelease rules",size=8.4)
    f.node(322,76,90,49,"Manifest\ntraining input",tone="CORAL",size=8.4)
    f.arrow([(357,193),(357,156),(244,156),(244,125)])
    f.arrow([(162,100),(200,100)],"GOLD")
    f.arrow([(288,100),(322,100)],"CORAL")
    f.line(18,169,412,169,"GOLD",dashed=True)
    f.text(18,177,"Record lineage at every transform",size=8.2,width=193)
    f.arrow([(90,169),(90,125)],"GOLD")
    f.footer("The manifest binds selected IDs and versions; payload objects stay immutable.")


def telemetry_pipeline(f):
    f.node(18,189,110,45,"Events + IDs\ntenant, time, offset")
    f.node(160,189,110,45,"Partition + dedup\nreplay contract",size=8.4)
    f.node(302,189,110,45,"Window state\nper-key partials",tone="GOLD")
    f.arrow([(128,211),(160,211)])
    f.arrow([(270,211),(302,211)])
    f.node(302,74,110,45,"Merge by key\nthen rank",tone="GOLD")
    f.node(160,74,110,45,"Versioned result\ncorrections / sink",tone="CORAL",size=8.4)
    f.node(18,74,110,45,"Recovery snapshot\nstate + offsets",size=8.4)
    f.arrow([(357,189),(357,119)],"GOLD")
    f.arrow([(302,96),(270,96)],"CORAL")
    f.text(18,156,"Checkpoint scope",bold=True,size=8.2)
    f.line(18,145,412,145,"TEAL",dashed=True)
    f.arrow([(73,145),(73,119)])
    f.footer("Summaries merge only under compatible keys, windows, hashes, and versions.",
             "Late data can revise results; recovery must align state with consumed offsets.")


def migration_gates(f):
    for i,lab in enumerate(["Shadow","Canary","New default","Migrate"]):
        f.node(18+i*101,158,91,38,lab,size=8.4)
        if i<3:f.arrow([(109+i*101,177),(119+i*101,177)])
    f.node(18,60,184,44,"At each commitment\nquality, SLOs, state compatibility",tone="GOLD",size=8.4)
    f.node(234,60,178,44,"Retire old path\nonly after verified exit criteria",tone="CORAL",size=8.4)
    f.arrow([(368,158),(368,104)],"CORAL")
    f.arrow([(64,158),(64,125),(110,125),(110,104)],"GOLD")
    f.footer("Define rollback for the migrated state, traffic cohort, and recovery time.")


FIGURES = {
    "token_alignment": (244, "NEXT-TOKEN SUPERVISION: ALIGN INPUTS AND TARGETS", token_alignment),
    "decoder_block": (297, "TWO RESIDUAL SUBLAYERS IN A DECODER BLOCK", decoder_block),
    "attention_state_map": (295, "WHAT HISTORY DOES A NEW QUERY READ?", attention_state_map),
    "optimization_loop": (270, "ONE OPTIMIZATION STEP, WITH PERSISTENT STATE", optimization_loop),
    "dedup_clusters": (254, "SIMILARITY EDGES BECOME DATASET GROUPS", dedup_clusters),
    "synthetic_gate": (242, "GENERATION BECOMES DATA ONLY AFTER VERIFICATION", synthetic_gate),
    "lora_paths": (255, "LOW-RANK ADAPTATION ADDS A TRAINABLE BRANCH", lora_paths),
    "policy_learning": (260, "FROM SAMPLED TRAJECTORIES TO A POLICY UPDATE", policy_learning),
    "kv_page_sharing": (281, "PREFIX REUSE SHARES PHYSICAL KV PAGES", kv_page_sharing),
    "kv_handoff": (240, "KV HANDOFF IS AN OWNERSHIP TRANSITION", kv_handoff),
    "quantization_path": (240, "LOW-BIT STORAGE MUST MATCH THE EXECUTION PATH", quantization_path),
    "double_buffer": (247, "TWO BUFFERS: OVERLAP WITH SAFE REUSE", double_buffer),
    "zero_shards": (281, "ZeRO: WHICH MODEL STATE DOES EACH RANK KEEP?", zero_shards),
    "phase_sharding": (326, "CONTEXT PARALLELISM CHANGES WITH THE PHASE", phase_sharding),
    "moe_dispatch": (300, "MoE: TOKENS TRAVEL TO SELECTED EXPERTS", moe_dispatch),
    "checkpoint_commit": (281, "CHECKPOINT SHARDS BECOME ONE COMMITTED STEP", checkpoint_commit),
    "event_time": (252, "EVENT TIME AND ARRIVAL ORDER ARE DIFFERENT AXES", event_time),
    "count_min": (276, "COUNT-MIN SKETCH: READ ONE COUNTER PER ROW", count_min),
    "rag_pipeline": (300, "RETRIEVAL-AUGMENTED GENERATION: TWO PATHS", rag_pipeline),
    "multimodal_path": (296, "MODALITIES ENTER THROUGH DIFFERENT ENCODERS", multimodal_path),
    "diffusion_blocks": (289, "BLOCK DIFFUSION: FIXED PREFIX, MUTABLE PRESENT", diffusion_blocks),
    "migration_gates": (250, "MIGRATION PROCEEDS THROUGH EVIDENCE GATES", migration_gates),
    "data_release": (294, "DATA RELEASES BIND PAYLOADS, LINEAGE, AND POLICY", data_release),
    "telemetry_pipeline": (290, "TELEMETRY: EVENTS BECOME RECOVERABLE WINDOW STATE", telemetry_pipeline),
}


def render_figure(name, width, fonts, palette):
    height, title, author = FIGURES[name]
    figure = Figure(height, title, fonts, palette)
    author(figure)
    return figure.finish(width)
