"""Fix broken training loop in generate_colab_notebook.py"""
with open('generate_colab_notebook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the broken batch_l list comprehension and the duplicate "cleaner training loop" comment
old = (
    "    tr_losses, v_losses = [], []\n"
    "    best_vl, best_state, best_ep, no_imp = float(\"inf\"), None, 0, 0\n"
    "\n"
    "    for ep in range(1, max_ep+1):\n"
    "        model.train()\n"
    "        batch_l = [crit(model(xb.to(DEVICE)), yb.to(DEVICE)).item()\n"
    "                   for xb, yb in tr_loader\n"
    "                   if not (model.zero_grad() or True)\n"
    "                   or (lambda: (opt.zero_grad(),\n"
    "                                (loss := crit(model(xb.to(DEVICE)), yb.to(DEVICE))),\n"
    "                                loss.backward(), opt.step(), loss.item()))()[-1]]\n"
    "\n"
    "        # cleaner training loop\n"
    "        model.train(); epoch_tr = []\n"
)

new = (
    "    tr_losses, v_losses = [], []\n"
    "    best_vl, best_state, best_ep, no_imp = float(\"inf\"), None, 0, 0\n"
    "\n"
    "    for ep in range(1, max_ep+1):\n"
    "        model.train(); epoch_tr = []\n"
)

if old in content:
    content = content.replace(old, new)
    with open('generate_colab_notebook.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed: removed broken batch_l comprehension.")
else:
    print("Pattern not found exactly. Trying line-by-line search...")
    # Find by key substring
    marker = "batch_l = [crit(model(xb.to(DEVICE))"
    if marker in content:
        lines = content.split('\n')
        new_lines = []
        skip = False
        for i, line in enumerate(lines):
            if 'batch_l = [crit' in line:
                skip = True
                continue
            if skip and '# cleaner training loop' in line:
                skip = False
                continue  # skip this comment too
            if not skip:
                new_lines.append(line)
        content = '\n'.join(new_lines)
        with open('generate_colab_notebook.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed via line-by-line scan.")
    else:
        print("batch_l marker not found either - file may already be clean")
