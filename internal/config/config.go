package config

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
)

type Config struct {
	Editor   string `json:"editor"`
	Terminal string `json:"terminal"`
	filePath string
}

func Load(path string) *Config {
	c := &Config{
		Editor:   "",
		Terminal: "",
		filePath: path,
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return c
	}

	if err := json.Unmarshal(data, c); err != nil {
		fmt.Fprintf(os.Stderr, "warning: config.json parse error: %v\n", err)
	}
	c.filePath = path
	return c
}

func (c *Config) Save() error {
	data, err := json.MarshalIndent(c, "", "    ")
	if err != nil {
		return err
	}
	return os.WriteFile(c.filePath, data, 0644)
}

// ResolveEditor returns the editor to use, checking config then $EDITOR then falling back to nano.
func (c *Config) ResolveEditor() string {
	if c.Editor != "" {
		return c.Editor
	}
	if env := os.Getenv("EDITOR"); env != "" {
		return env
	}
	return "nano"
}

// ResolveTerminal returns the terminal emulator to use.
// Checks config, then $TERMINAL, then probes common terminal emulators.
func (c *Config) ResolveTerminal() string {
	if c.Terminal != "" {
		return c.Terminal
	}
	if env := os.Getenv("TERMINAL"); env != "" {
		return env
	}

	// Probe common terminal emulators in preference order
	candidates := []string{
		"kitty",
		"alacritty",
		"wezterm",
		"gnome-terminal",
		"konsole",
		"xfce4-terminal",
		"xterm",
	}
	for _, t := range candidates {
		if _, err := exec.LookPath(t); err == nil {
			return t
		}
	}
	return "xterm"
}
