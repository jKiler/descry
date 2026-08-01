package lang

// Prose and data files carry no grammar: they exist in the registry so the set
// of indexed extensions lives in exactly one place, and so their (nonexistent)
// test conventions are stated rather than assumed. They chunk through the
// fallback chunker, which is the right unit for them — a markdown section or a
// blank-line-delimited paragraph.
var (
	Markdown = &Pack{
		Name: "markdown",
		Exts: []string{".md", ".markdown"},
	}
	Text = &Pack{
		Name: "text",
		Exts: []string{".txt", ".rst"},
	}
	Data = &Pack{
		Name:             "data",
		Exts:             []string{".json", ".yaml", ".yml"},
		GeneratedMarkers: []string{"DO NOT EDIT", "@generated"},
	}
)

func init() {
	Register(Markdown)
	Register(Text)
	Register(Data)
}
