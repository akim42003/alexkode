package main

import (
	"fmt"
	tea "github.com/charmbracelet/bubbletea"
	"os"
)

type problem_tree struct {
	category   []string
	completed  []bool
	difficulty []string
}

type model struct {
	currentPage  string
	select_type  []string
	cursor       int
	selected     map[int]struct{}
	problem_tree problem_tree
}

func initialModel() model {
	return model{
		currentPage: "select_problem",
		select_type: []string{"category", "completed", "difficulty"},
		cursor:      0,
		selected:    make(map[int]struct{}),
		problem_tree: problem_tree{
			category:   []string{"Linear Algebra", "Statistics and Probability", "Machine Learning", "Deep Learning", "NLP", "Computer Vision"},
			completed:  []bool{true, false},
			difficulty: []string{"Easy", "Medium", "Hard"},
		},
	}

}

func (m model) Init() tea.Cmd {
	return nil
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {

	case tea.KeyMsg:

		switch msg.String() {

		case "ctrl+c", "q":
			return m, tea.Quit

		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			}

		case "down", "j":
			if m.cursor < len(m.select_type)-1 {
				m.cursor++
			}
		case "enter", " ":

			if m.currentPage == "select_problem" {
				_, ok := m.selected[m.cursor]
				if ok {
					delete(m.selected, m.cursor)
				} else {
					m.selected[m.cursor] = struct{}{}
				}
			}
		case "c":
			if m.currentPage == "select_problem" && len(m.selected) > 0 {
				m.currentPage = "select_id"
			}
		case "b":
			if m.currentPage == "select_id" {
				m.currentPage = "select_problem"
			}
		}
	}

	return m, nil

}

func (m model) View() string {
	switch m.currentPage {

	case "select_problem":
		return m.renderSelectionPage()
	case "select_id":
		return m.renderProblemsPage()
	default:
		return "Unknown page"

	}
}

func main() {
	p := tea.NewProgram(initialModel())
	if _, err := p.Run(); err != nil {
		fmt.Printf("error", err)
		os.Exit(1)
	}

}
