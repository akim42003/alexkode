package main

import (
	"fmt"
	"os"
	"path/filepath"

	"v0/internal/tui"

	tea "github.com/charmbracelet/bubbletea"
)

func main() {
	exe, err := os.Executable()
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	baseDir := filepath.Dir(exe)

	// Allow override via environment
	if dir := os.Getenv("DEEPML_DIR"); dir != "" {
		baseDir = dir
	}

	// Also check if running from source (go run)
	if _, err := os.Stat(filepath.Join(baseDir, "problems")); os.IsNotExist(err) {
		// Try current working directory
		cwd, _ := os.Getwd()
		if _, err := os.Stat(filepath.Join(cwd, "problems")); err == nil {
			baseDir = cwd
		}
	}

	problemsDir := filepath.Join(baseDir, "problems")
	userCodeDir := filepath.Join(baseDir, "user_code")
	progressPath := filepath.Join(baseDir, "progress.json")
	configPath := filepath.Join(baseDir, "config.json")

	model := tui.NewModel(problemsDir, userCodeDir, progressPath, configPath)

	p := tea.NewProgram(model, tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
