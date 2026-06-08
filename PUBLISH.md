# Publishing to GitHub

The local repository is ready to publish. Authenticate with GitHub first:

```bash
gh auth login -h github.com
```

Then create a new repository under `cau-hai-lab` and push:

```bash
cd /data2/AAAI/hai_mw/projects/ECSO/release/ecso-gptq-mllmjailbreak-ko
gh repo create cau-hai-lab/ecso-gptq-mllmjailbreak-ko \
  --private \
  --source=. \
  --remote=origin \
  --push
```

Use `--public` instead of `--private` only after confirming that the organization wants the reports and metadata to be public.

## Publishing Model Weights

The current initial commit does not include the 15 GB GPTQ `safetensors` checkpoint. To publish the checkpoint through Git LFS:

```bash
git lfs install
git lfs track "*.safetensors"
mkdir -p gptq_/outputs/axvl_gptq_llm_w4g128_v2
cp /path/to/axvl_gptq_llm_w4g128_v2/*.safetensors gptq_/outputs/axvl_gptq_llm_w4g128_v2/
cp weights/axvl_gptq_llm_w4g128_v2_metadata/* gptq_/outputs/axvl_gptq_llm_w4g128_v2/
git add .gitattributes gptq_/outputs/axvl_gptq_llm_w4g128_v2
git commit -m "Add GPTQ TELL checkpoint with Git LFS"
git push
```

Check the organization's Git LFS quota before pushing the checkpoint.

