package main

import (
	"reflect"
	"testing"
)

func TestParseArgs(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want invocation
	}{
		{"bare", nil, invocation{cmd: cmdHome}},
		{"query joins words", []string{"how", "is", "auth", "implemented"},
			invocation{cmd: cmdQuery, query: "how is auth implemented"}},
		{"quoted query", []string{"where is auth handled"},
			invocation{cmd: cmdQuery, query: "where is auth handled"}},
		{"subcommand", []string{"index", "some/dir"},
			invocation{cmd: "index", args: []string{"some/dir"}}},
		{"search is reserved", []string{"search", "index"},
			invocation{cmd: "search", args: []string{"index"}}},
		{"single reserved word is a subcommand, not a query", []string{"status"},
			invocation{cmd: "status", args: []string{}}},
		{"help word", []string{"help"}, invocation{cmd: cmdHelp}},
		{"help flag", []string{"--help"}, invocation{cmd: cmdHelp}},
		{"short help flag", []string{"-h"}, invocation{cmd: cmdHelp}},
		{"unknown flag is usage, not a query", []string{"--verbose", "auth"},
			invocation{cmd: cmdUsage}},
		{"blank query is usage", []string{"  "}, invocation{cmd: cmdUsage}},
		{"reserved word later in a query stays a query", []string{"how", "does", "index", "work"},
			invocation{cmd: cmdQuery, query: "how does index work"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := parseArgs(tc.args)
			// Normalize nil vs empty for comparison.
			if len(got.args) == 0 {
				got.args = nil
			}
			if len(tc.want.args) == 0 {
				tc.want.args = nil
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("parseArgs(%q) = %+v, want %+v", tc.args, got, tc.want)
			}
		})
	}
}

// Every name in the subcommands set must stay dispatchable, and the set is the
// single place that grows when a command is added — this test is the reminder
// that each addition is a breaking change for one-word queries (see the
// dispatch comment in main.go).
func TestReservedWordsDispatchAsSubcommands(t *testing.T) {
	for name := range subcommands {
		if got := parseArgs([]string{name}); got.cmd != name {
			t.Errorf("parseArgs(%q).cmd = %q, want the subcommand itself", name, got.cmd)
		}
	}
}
