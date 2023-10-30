import torch
import torch.nn as nn
from torch.nn import functional as F

# Sorry for the terrible german commenting, but this helps me to understand what the whole thing is about
# This "documentation" works best in combination with 'nanoGPT_Start.ipynb', where those comments originate from

# hyperparameters
batch_size = 64 # Menge der parallel bearbeiteten Blöcke
block_size = 256 # Maximale Kontext-Länge
max_iters = 5000 # Trainingsiterationen
eval_interval = 500 # Evaluationsintervall für Loss-Durchschnitt
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' # GGF auf GPU laufen, erfordert dann dieses ganze .to(device)
eval_iters = 200 # Menge der Samples für Loss-Durchschnitt, damit die Approximation genauer + stabiler ist
n_embd = 32 # "Number of embedding dimensions"
n_embd = 384 # Embedding Dimensions für Tabellen / Tensoren
n_head = 6 # Wie viele Heads für Attention (384/6 = 64)
n_layer = 6 # Layers für Attention -> Gleicher Grund wie Oben
dropout = 0.2 # Overfitting-Prevention, indem random Kommunikationswege unterbunden werden
# ------------

# Seed für Random-Gedöns -> Wichtig für Reproduzierbarkeit
torch.manual_seed(2424)

# Datensatz "tiny-shakespeare" laden und einlesen
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Vokabular ermitteln
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Chars auf Ints mappen und umgekehrt (als Liste)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
# Daraus dann Encoding und Decoding
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Trainings- und Testdaten splitten
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # Hier 90% Trainingsdaten
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # Je nachdem, ob Training oder Validation, zufällige Sequenz auswählen
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Die Tensoren füllen (Einmal Context, einmal Targets)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y

@torch.no_grad() # Alles was hier passiert, muss nicht .backward fähig sein, also cache sparen!
def estimate_loss():
    # Hier wird der Loss-Durchschnitt berechnet
    out = {}
    model.eval() # Das hat hier eigentlich noch keinen Sinn, erinnert aber an die richtige Moduseinstellung des Modells
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Denn je nach Modus kann sich das Modell ggf. anders verhalten
    return out


# Jetzt gehts los mit der self-attention
class Head(nn.Module):
    # Mit einem Head -> Mulit Head kommt später und ist noch bissl anders

    def __init__(self, head_size):
        super().__init__()
        # Aufsetzen der Key, Query und Value Linear NNs wie im Notebook
        self.key = nn.Linear(n_embd, head_size, bias=False) # Was ein Token beinhaltet
        self.query = nn.Linear(n_embd, head_size, bias=False) # Wonach ein Token sucht
        self.value = nn.Linear(n_embd, head_size, bias=False) # Rohdaten-Dimensionsredukltion auf Head-Size
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # Die Dreiecksmatrix mit Einsen

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # Neue Smarte Affinität berechnen
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T), damit Zukunft nicht bedacht wird
        wei = F.softmax(wei, dim=-1) # (B, T, T), Normalisieren
        wei = self.dropout(wei)

        # Input mit Affinität verrechnen
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    # Wenn man mehrere Heads benutzt, kann man dadurch parallel bessere Ergebnisse erziehlen
    # Dafür "einfach" parallel ausführen und am ende konkatenieren

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # Für residual connections
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# FeedForward Komponente, um die Auswahl des richtigen Tokens zu verbessern, indem auf den self-attention Daten
# nochmal per Token gerechnet wird
class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # Diese Struktur hat mit residual connections und guten Ansatzwerten aus dem Attention-Paper zu tun
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    # Transformer-Block -> Wendet jetzt genau dieses Prinzip an: Erst self attention (communication), dann computation

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        # Hier sind jetzt die Mulit-Heads und das FFWD-NN
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # Layer Normalization als verbesserung von Batch-Normalization
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # x + ... ist smart, weil damit diese residual connections genutzt werden
        x = x + self.ffwd(self.ln2(x)) # Neu: Pre-Normalization, anstatt wie im Paper Post-Normalization
        return x


# Einfachstes Beispiel -> BigramLanguageModel
# Idee: Lernen, welche Buchstaben direkt aufeinander folgen
# Also zum Beispiel: Wenn K, dann vermutlich a
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Aufbau einer Tabelle, in der die Aufeinanderfolgewahrscheinlichkeit steht (jetzt mit Embedding Dimensions)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Aufbau einer weiteren Tabelle, in der die Positionen der Tokens gespeichert sind
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Linear Layer, der daraus dann die Logits berechnet (Head für Language Model)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Transformer-Blöcke definieren
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm

    def forward(self, idx, targets=None):

        # B und T extrahieren für Funktionen
        B, T = idx.shape

        # Aufbau eines Tensors, der für jeden Batch die Blöcke mit den möglichen
        # Werten der Tabelle fusioniert, sodass am Ende (Batch x Times x Channels)
        # also 4 x 8 x 65 als Tensor entsteht -> Damit dann vorhersagen, für
        # welchen Char, welcher char als nächstes Kommt

        # Komplexer mit n_embd -> Erst die Embedding tables aufbauen
        token_emb = self.token_embedding_table(idx) # Identität / Affinität der Tokens
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # Position der Tokens im Block
        x = token_emb + pos_emb # Fusion aus Position und Identität
        x = self.blocks(x) # FFWD und Head-Computation in Blocks ausgelagert
        logits = self.lm_head(x) # Das hier ist jetzt B, T, C wie oben erklärt, nach Durchlauf des Linear Layers

        if targets is None:
            loss = None
        else:
            # Und weil (warum auch immer) das zunächst die falsche Form/Dimension hat:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # Jetzt noch den Loss berechnen, um den Tensor zu updaten
            # Der Loss ist übrigens der Grund für den Dimensionskäse oben
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # Jetzt kann man mit dem Modell auch etwas genereiren lassen
    def generate(self, idx, max_new_tokens):
        # idx ist das B x T Array aus Indices der Tokens
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # idx croppen, weil die Position-Embedding-Table sond Out of Scope geht
            # Prediction vom Modell holen
            logits, loss = self(idx_cond)
            # Fokus nur auf den letzten char (weil BigramLanguageModel)
            logits = logits[:, -1, :]  # Nur noch B x C
            # Softmax nutzen, um Wahrscheinlichkeit zu ermitteln
            probs = F.softmax(logits, dim=-1)
            # Aus der ermittelten Verteilung ziehen
            idx_next = torch.multinomial(probs, num_samples=1)  # Wird B x 1
            # Den ermittelten Int eines Chars an die Sequenz anhängen
            idx = torch.cat((idx, idx_next), dim=1)  # Wird B X T+1
        return idx

model = BigramLanguageModel()
m = model.to(device)

# Jetzt gehts um basic learning, damit das Teil auch mal was rallt

# Optimizer erstellen
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # Hier findet in oben defineirten Abständen die neue Loss-Ausgabe statt, die den Durchschnittswert nutzt
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Batch ziehen
    xb, yb = get_batch('train')

    # Loss berechnen und optimieren
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Nach dem Training eine Sequenz aus 500 chars generieren
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))