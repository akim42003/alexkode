package main

import (
	"fmt"
)

func (m model) renderSelectionPage() string {

	s := "How would you like to select a problem?\n"

	for i, choice := range m.select_type {
		cursor := " "
		if m.cursor == i {
			cursor = ">"
		}
		checked := " "
		if _, ok := m.selected[i]; ok {
			checked = "x"
		}

		s += fmt.Sprintf("%s [%s] %s\n", cursor, checked, choice)
	}
	s += "\nPress 'b' to go back | Press 'q' to quit\n"

	return s
}
