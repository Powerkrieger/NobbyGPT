import torch
import torch.nn as nn
from torch.nn import functional as F

# Sorry for the terrible german commenting, but this helps me to understand what the whole thing is about
# This "documentation" works best in combination with 'nanoGPT_Start.ipynb', where those comments originate from

# hyperparameters
batch_size = 32 # Menge der parallel bearbeiteten Blöcke
block_size = 8 # Maximale Kontext-Länge
max_iters = 3000 # Trainingsiterationen
eval_interval = 300 # Evaluationsintervall für Loss-Durchschnitt
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu' # GGF auf GPU laufen, erfordert dann dieses ganze .to(device)
eval_iters = 200 # Menge der Samples für Loss-Durchschnitt, damit die Approximation genauer + stabiler ist
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

# Einfachstes Beispiel -> BigramLanguageModel
# Idee: Lernen, welche Buchstaben direkt aufeinander folgen
# Also zum Beispiel: Wenn K, dann vermutlich a
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Aufbau einer Tabelle, in der die Aufeinanderfolgewahrscheinlichkeit steht
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # Aufbau eines Tensors, der für jeden Batch die Blöcke mit den möglichen
        # Werten der Tabelle fusioniert, sodass am Ende (Batch x Times x Channels)
        # also 4 x 8 x 65 als Tensor entsteht -> Damit dann vorhersagen, für
        # welchen Char, welcher char als nächstes Kommt
        logits = self.token_embedding_table(idx)

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
            # Prediction vom Modell holen
            logits, loss = self(idx)
            # Fokus nur auf den letzten char (weil BigramLanguageModel)
            logits = logits[:, -1, :] # Nur noch B x C
            # Softmax nutzen, um Wahrscheinlichkeit zu ermitteln
            probs = F.softmax(logits, dim=-1)
            # Aus der ermittelten Verteilung ziehen
            idx_next = torch.multinomial(probs, num_samples=1) # Wird B x 1
            # Den ermittelten Int eines Chars an die Sequenz anhängen
            idx = torch.cat((idx, idx_next), dim=1) # Wird B X T+1
        return idx

model = BigramLanguageModel(vocab_size)
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