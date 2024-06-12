from src.train import train

train(
    subject="UTS01",
    model="sentence-transformers/all-mpnet-base-v2",
    decoder="brain_decoder",
    context_length=4,
    lag=2,
    batch_size=1024,
)
