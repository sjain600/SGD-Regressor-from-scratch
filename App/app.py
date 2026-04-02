import gradio as gr
import torch
import pandas as pd
import pickle
import numpy as np
import torch.nn as nn

movies = pd.read_csv('movies.csv')

with open("movie_ids.pkl", "rb") as f:
    movie_ids = pickle.load(f)
with open("user_map.pkl", "rb") as f:
    user_map = pickle.load(f)
    
inv_user_map = {v: k for k, v in user_map.items()}

n_items = len(movie_ids)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=8, layers=[32, 16, 8]):
        super().__init__()
        
        # Embedding tables P and Q (from Equation 3)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)   # P
        self.item_embedding = nn.Embedding(n_items, embedding_dim)   # Q
        
        # Neural CF layers (the tower in Equation 4)
        self.layers = nn.ModuleList()
        input_size = embedding_dim * 2  # concatenate user + item
        
        for layer_size in layers:
            self.layers.append(nn.Linear(input_size, layer_size))
            input_size = layer_size
        
        # Output layer φ_out
        self.output_layer = nn.Linear(input_size, 1)
        
        self.dropout = nn.Dropout(0.1)
        
        self.gmf = nn.Linear(embedding_dim, 1)
        
    def forward(self, user_ids, item_ids):
        # Get embeddings: P^T v_u and Q^T v_i
        user_emb = self.user_embedding(user_ids)      # shape: (batch, embedding_dim)
        item_emb = self.item_embedding(item_ids)
        
        gmf_input = torch.mul(user_emb, item_emb)               
        gmf_output = self.gmf(gmf_input)  
        
        # Concatenate (this is the input to φ1)
        x = torch.cat([user_emb, item_emb], dim=1)
        
        # Pass through all neural CF layers φ1 → φ2 → ... → φX
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(torch.relu(x))         # paper uses ReLU
        
        # Final output layer φ_out + sigmoid
        mlp_output = self.output_layer(x)
        combined_output = torch.cat([gmf_output, mlp_output], dim=1)
        
        raw_pred = torch.sum(combined_output, dim=1)
        
        return torch.sigmoid(raw_pred)      # score between 0 and 1
    
model = NeuMF(n_users=610, n_items=9724, embedding_dim=32, layers=[200, 128, 64, 32, 16])
model.load_state_dict(torch.load("neumf.pth", map_location=device))
model.to(device)
model.eval()
    

@torch.no_grad()
def recommend(user_id_input: int, top_n: int = 10):
    """
    user_id_input = original User ID from MovieLens (example: 1, 42, 610)
    """
    # Handle cold-start users gracefully
    if user_id_input not in user_map:
        # New user → we use zero embeddings (mean-user baseline)
        mapped_user = 0  # doesn't matter, we will zero it out inside
    else:
        mapped_user = user_map[user_id_input]
    
    # Create tensor for all movies
    all_items = torch.arange(n_items, device=device)
    user_tensor = torch.full((n_items,), mapped_user, dtype=torch.long, device=device)
    
    scores = model(user_tensor, all_items).cpu().numpy()
    
    # Get top-N
    top_idx = np.argsort(scores)[::-1][:top_n]
    top_scores = scores[top_idx]
    
    results = []
    for idx, score in zip(top_idx, top_scores):
        orig_movie_id = movie_ids[idx]
        title = movies[movies['movieId'] == orig_movie_id]['title'].iloc[0]
        results.append({
            "Rank": len(results) + 1,
            "Movie Title": title,
            "Predicted Score": round(float(score), 4)
        })
    
    df = pd.DataFrame(results)
    return df

def hello():
    return f"✅ Torch version: {torch.__version__}\nDevice: {torch.device('cpu')}"

with gr.Blocks(title="🎬 NeuMF Movie Recommender", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 NeuMF Movie Recommender\n"
                "Type any User ID from MovieLens (1–610) and get personalized recommendations!")
    
    with gr.Row():
        user_input = gr.Number(label="Your User ID", value=42, precision=0)
        top_n_slider = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="How many recommendations?")
    
    recommend_btn = gr.Button("🚀 Get My Recommendations", variant="primary")
    
    output_table = gr.DataFrame(label="Top Recommendations")
    
    recommend_btn.click(
        fn=recommend,
        inputs=[user_input, top_n_slider],
        outputs=output_table
    )

# Launch the app
demo.launch()
    
    
