import safetensors.torch as st

sd = st.load_file("/workspace/asset_generator/out_lora/char_lora.safetensors")
for i, k in enumerate(sd.keys()):
    print(i, k)
    if i > 50:
        break
