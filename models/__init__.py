from .model_apuSmog import build_apuSmog

MODEL_FUNCS = {
  "apu_smog": build_apuSmog,
}


def build_model(args):
  model = MODEL_FUNCS[args.model_name](args)
  return model
