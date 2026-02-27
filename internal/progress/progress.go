package progress

import (
	"encoding/json"
	"os"
	"sync"
)

type Progress struct {
	Completed  map[string]bool `json:"completed"`
	LastOpened string          `json:"last_opened"`
	mu         sync.Mutex
	filePath   string
}

func Load(path string) *Progress {
	p := &Progress{
		Completed: make(map[string]bool),
		filePath:  path,
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return p
	}

	var raw struct {
		Completed  []string `json:"completed"`
		LastOpened string   `json:"last_opened"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return p
	}

	for _, id := range raw.Completed {
		p.Completed[id] = true
	}
	p.LastOpened = raw.LastOpened
	return p
}

func (p *Progress) Save() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	var completed []string
	for id := range p.Completed {
		completed = append(completed, id)
	}

	raw := struct {
		Completed  []string `json:"completed"`
		LastOpened string   `json:"last_opened"`
	}{
		Completed:  completed,
		LastOpened: p.LastOpened,
	}

	data, err := json.MarshalIndent(raw, "", "    ")
	if err != nil {
		return err
	}

	return os.WriteFile(p.filePath, data, 0644)
}

func (p *Progress) MarkCompleted(problemID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.Completed[problemID] = true
}

func (p *Progress) MarkIncomplete(problemID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.Completed, problemID)
}

func (p *Progress) IsCompleted(problemID string) bool {
	return p.Completed[problemID]
}

func (p *Progress) SetLastOpened(problemID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.LastOpened = problemID
}

func (p *Progress) CompletedCount() int {
	return len(p.Completed)
}

func (p *Progress) CompletedInCategory(categorySlug string, problems []string) int {
	count := 0
	for _, id := range problems {
		if p.Completed[id] {
			count++
		}
	}
	return count
}
