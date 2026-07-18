# ✅ BIEN — mismo encoder para referencia y candidato
from openai import OpenAI
client = OpenAI()
ref_emb = client.embeddings.create(input=ref, model="text-embedding-3-small").data[0].embedding
# (en producción, compararías con un candidato ya embedido por el mismo modelo)

# ❌ MAL — referencia con OpenAI, candidato con Cohere
# Los cosenos no son comparables aunque la fórmula lo permita