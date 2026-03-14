package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

const (
	apiBase = "https://site--bot--dxfjds728w5v.code.run"
	cliID   = "081e2347-26b8-426b-bde9-5ccb03833e7a"
	gpu     = "B200_Nebius"
	topN    = 10
)

var leaderboards = [4]string{
	"causal_conv1d",
	"gated_deltanet_chunk_fwd_h",
	"gated_deltanet_chunk_fwd_o",
	"gated_deltanet_recompute_w_u",
}

var shortNames = [4]string{
	"Causal Conv1D",
	"Chunk Fwd H",
	"Chunk Fwd O",
	"Recompute W/U",
}

var (
	white   = lipgloss.Color("#F5F5F7")
	light   = lipgloss.Color("#D1D1D6")
	dimGray = lipgloss.Color("#48484A")
	border  = lipgloss.Color("#2C2C2E")
	red     = lipgloss.Color("#FF453A")
	blue    = lipgloss.Color("#0A84FF")
)

type entry struct {
	Rank  int     `json:"rank"`
	User  string  `json:"user_name"`
	Score float64 `json:"submission_score"`
}

type boardData struct {
	idx     int
	entries []entry
	err     error
}

type tickMsg time.Time

type model struct {
	boards    [4][]entry
	errors    [4]error
	lastFetch time.Time
	width     int
	height    int
}

func formatTime(seconds float64) string {
	switch {
	case seconds >= 1:
		return fmt.Sprintf("%6.2fs", seconds)
	case seconds >= 1e-3:
		return fmt.Sprintf("%6.2fms", seconds*1e3)
	case seconds >= 1e-6:
		return fmt.Sprintf("%6.2fµs", seconds*1e6)
	default:
		return fmt.Sprintf("%6.2fns", seconds*1e9)
	}
}

func fetchBoard(idx int) tea.Cmd {
	return func() tea.Msg {
		url := fmt.Sprintf("%s/submissions/%s/%s", apiBase, leaderboards[idx], gpu)
		req, _ := http.NewRequest("GET", url, nil)
		req.Header.Set("X-Popcorn-Cli-Id", cliID)

		resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
		if err != nil {
			return boardData{idx: idx, err: err}
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return boardData{idx: idx, err: fmt.Errorf("HTTP %d", resp.StatusCode)}
		}

		body, _ := io.ReadAll(resp.Body)
		var entries []entry
		if err := json.Unmarshal(body, &entries); err != nil {
			return boardData{idx: idx, err: err}
		}
		if len(entries) > topN {
			entries = entries[:topN]
		}
		return boardData{idx: idx, entries: entries}
	}
}

func fetchAll() tea.Cmd {
	cmds := make([]tea.Cmd, 4)
	for i := range cmds {
		cmds[i] = fetchBoard(i)
	}
	return tea.Batch(cmds...)
}

func tickCmd() tea.Cmd {
	return tea.Tick(5*time.Second, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func (m model) Init() tea.Cmd {
	return tea.Batch(fetchAll(), tickCmd())
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c", "esc":
			return m, tea.Quit
		case "r":
			m.lastFetch = time.Now()
			return m, fetchAll()
		}
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
	case boardData:
		if msg.err == nil {
			m.boards[msg.idx] = msg.entries
			m.errors[msg.idx] = nil
			m.lastFetch = time.Now()
		} else if m.boards[msg.idx] == nil {
			// Only show error if we have no prior data
			m.errors[msg.idx] = msg.err
		}
		// If we have prior data and got an error, silently keep stale data
	case tickMsg:
		return m, tea.Batch(fetchAll(), tickCmd())
	}
	return m, nil
}

func (m model) View() string {
	if m.width == 0 {
		return ""
	}

	colWidth := m.width / 4
	if colWidth < 32 {
		colWidth = 32
	}
	innerW := colWidth - 6

	// Styles
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(white).
		Width(innerW).
		AlignHorizontal(lipgloss.Center)

	dimStyle := lipgloss.NewStyle().
		Foreground(dimGray)

	yhinaiStyle := lipgloss.NewStyle().
		Foreground(blue).
		Bold(true)

	ramizzikStyle := lipgloss.NewStyle().
		Foreground(red).
		Bold(true)

	rowStyle := lipgloss.NewStyle().
		Foreground(light)

	colStyle := lipgloss.NewStyle().
		Width(colWidth).
		Padding(1, 3).
		Border(lipgloss.NormalBorder(), false, true, false, false).
		BorderForeground(border)

	lastColStyle := colStyle.Border(lipgloss.HiddenBorder(), false, true, false, false)

	columns := make([]string, 4)
	for i := 0; i < 4; i++ {
		var sb strings.Builder

		// Column title
		sb.WriteString(titleStyle.Render(shortNames[i]))
		sb.WriteString("\n")
		sb.WriteString(dimStyle.Render(strings.Repeat("─", innerW)))
		sb.WriteString("\n\n")

		if m.errors[i] != nil {
			sb.WriteString(dimStyle.Render("error"))
		} else if m.boards[i] == nil {
			sb.WriteString(dimStyle.Render("..."))
		} else if len(m.boards[i]) == 0 {
			sb.WriteString(dimStyle.Render("no entries"))
		} else {
			for _, e := range m.boards[i] {
				user := e.User
				if len(user) > 12 {
					user = user[:11] + "~"
				}

				rank := fmt.Sprintf("%2d", e.Rank)
				score := formatTime(e.Score)
				line := fmt.Sprintf("%s  %-12s %s", rank, user, score)

				lower := strings.ToLower(e.User)
				if strings.Contains(lower, "yhinai") {
					sb.WriteString(yhinaiStyle.Render(line))
				} else if strings.Contains(lower, "ramizzik") {
					sb.WriteString(ramizzikStyle.Render(line))
				} else {
					sb.WriteString(rowStyle.Render(line))
				}
				sb.WriteString("\n")
			}
		}

		// Pad to align columns
		lines := strings.Count(sb.String(), "\n")
		target := topN + 6
		for lines < target {
			sb.WriteString("\n")
			lines++
		}

		if i == 3 {
			columns[i] = lastColStyle.Render(sb.String())
		} else {
			columns[i] = colStyle.Render(sb.String())
		}
	}

	// Title
	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(white).
		Width(m.width).
		AlignHorizontal(lipgloss.Center).
		Render("Helion Hackathon")

	// Board
	board := lipgloss.JoinHorizontal(lipgloss.Top, columns[:]...)

	// Footer
	ts := "..."
	if !m.lastFetch.IsZero() {
		ts = m.lastFetch.Format("15:04:05")
	}
	footer := lipgloss.NewStyle().
		Foreground(dimGray).
		Width(m.width).
		AlignHorizontal(lipgloss.Center).
		Render(fmt.Sprintf("%s   r refresh   q quit", ts))

	return "\n" + title + "\n\n" + board + "\n\n" + footer + "\n"
}

func main() {
	p := tea.NewProgram(model{}, tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
