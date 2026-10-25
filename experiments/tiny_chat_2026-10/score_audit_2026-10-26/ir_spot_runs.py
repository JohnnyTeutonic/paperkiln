TC = r"C:/ml_artifacts/tiny_chat"
RUNS = {"ctl_r1": (f"{TC}/e1_x5_ctl/post_e1_x5_ctl_L16_r1", f"{TC}/e1/pick_e1_L16/model.gguf"),
        "x5_r1": (f"{TC}/e1_x5_r1/post_e1_x5_X5_r1", f"{TC}/e1_x5_r1/post_e1_x5_X5_r1/sft/tc_sft_X5_r1.gguf"),
        "x5_r2": (f"{TC}/e1_x5_r2/post_e1_x5_X5_r2", f"{TC}/e1_x5_r2/post_e1_x5_X5_r2/sft/tc_sft_X5_r2.gguf"),
        "x5_r3": (f"{TC}/e1_x5_r3/post_e1_x5_X5_r3", f"{TC}/e1_x5_r3/post_e1_x5_X5_r3/sft/tc_sft_X5_r3.gguf")}
