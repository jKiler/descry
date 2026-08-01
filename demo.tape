# demo.tape — source for the README demo gif. Regenerate with `make demo`
# (needs vhs: https://github.com/charmbracelet/vhs, `brew install vhs`).
#
# The hidden setup extracts a pristine `git archive` copy of this repo into
# /tmp/descry (a short path, so the index-consent prompt reads well) and puts
# the freshly built ./descry on PATH — the recording never shows untracked
# local files and reproduces from any checkout. The recording then shows the
# real cold start: the query prompts for consent, indexes with live progress,
# and answers; `descry doctor` then reports the whole resolution chain.
#
# Honest-timing caveat: the machine recording this already has the embedding
# model and ONNX Runtime cached at user level. A true first run downloads the
# ~90MB model once before indexing; the gif would be minutes long if it showed
# that, so it starts from "model cached, repo not indexed".

Output demo.gif

Set Shell bash
Set FontSize 30
Set Width 1600
Set Height 1080
Set Padding 40
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

Hide
Type `export PATH="$PWD:$PATH" DESCRY_CHUNKER=ast-go && rm -rf /tmp/descry && mkdir /tmp/descry && git archive HEAD | tar -x -C /tmp/descry && cd /tmp/descry && clear`
Enter
Wait
Show

# Cold start: no index yet — descry asks before touching anything.
Type `descry "where do vector and keyword results get fused into one ranking"`
Sleep 600ms
Enter
Sleep 2s
Type "y"
Sleep 400ms
Enter
# Indexing runs with live progress, then the answer prints.
Wait@2m
Sleep 3.5s

# The resolution chain, reported at once: cgo, runtime, model, index, skill.
Type `descry doctor`
Sleep 600ms
Enter
Sleep 5s
