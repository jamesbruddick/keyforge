//! Terminal presentation: colours, alignment, and progress.
//!
//! Every line a run prints goes through here, so the whole tool reads as one
//! table: a right-aligned label in a fixed gutter, then a value at a shared
//! column. Colour is dropped automatically when the stream is redirected or
//! `NO_COLOR` is set, so a piped run or a multi-day log stays plain text.

use std::io::{IsTerminal, Write};
use std::sync::Mutex;

/// Labels sit right-aligned in a gutter this wide, so every value in the run
/// starts at the same column.
const GUTTER: usize = 12;

/// Column where values begin: two-space margin, gutter, two-space separator.
const VALUE_COL: usize = 2 + GUTTER + 2;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";

const HIDE_CURSOR: &str = "\x1b[?25l";
const SHOW_CURSOR: &str = "\x1b[?25h";

/// Return to column 0 and erase the row.
const ERASE_ROW: &str = "\r\x1b[K";
/// Move the cursor up one row, staying in the same column.
const CURSOR_UP: &str = "\x1b[A";

/// One colour of the palette, in both the depths a terminal might offer.
///
/// The index is the nearest xterm-256 slot, for a terminal that never
/// advertised truecolor. Naming both keeps the approximation next to the colour
/// it stands in for rather than in a conversion that would have to guess.
struct Color {
    rgb: (u8, u8, u8),
    index: u8,
}

impl Color {
    const fn new(rgb: (u8, u8, u8), index: u8) -> Self {
        Self { rgb, index }
    }
}

/// The palette, named by the role each colour plays rather than by the colour it
/// currently is -- so re-theming the tool is a change to these five values and
/// nothing else.
///
/// Violet is the register security tooling is read in, and this is a CVE scanner
/// rather than a Bitcoin product: borrowing Bitcoin's orange would imply an
/// endorsement the tool does not have. Magenta is reserved for a filter hit,
/// because it is the one line in a multi-day sweep that must not be scrolled
/// past, and it is the only warm colour in the set apart from the two that mean
/// something is wrong. Lavender is quiet enough to carry addresses and phrases,
/// which are long and everywhere.
/// Cyan is the odd one out on purpose: an interrupt is neither a result nor a
/// fault, so it should not borrow the colour of either. Being the only cool
/// accent in a violet set is what makes it findable when it lands mid-sweep.
const HEADLINE: Color = Color::new((0xa7, 0x8b, 0xfa), 141);
const DATA: Color = Color::new((0xdd, 0xd6, 0xfe), 189);
const FOUND: Color = Color::new((0xf4, 0x72, 0xb6), 205);
const NOTICE: Color = Color::new((0x22, 0xd3, 0xee), 45);
const WARNING: Color = Color::new((0xfb, 0xbf, 0x24), 214);
const FAILURE: Color = Color::new((0xf8, 0x71, 0x71), 210);

/// Which stream a subcommand's status lines belong on.
///
/// `verify` and `bench` print a result, which belongs on stdout where it can be
/// piped. `scan`'s real output is the matches file, so everything it prints is
/// status and goes to stderr -- that keeps `scan ... 2> scan.log` a complete log
/// and leaves stdout free.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Stdout,
    Stderr,
}

pub struct Ui {
    stream: Stream,
    /// Whether the row stream takes colour.
    color: bool,
    /// Progress is only drawn in place when stderr is a real terminal.
    interactive: bool,
    /// Whether the terminal said it can take 24-bit colour. The violets sit in a
    /// part of the space the 256-colour cube samples coarsely, so it is worth
    /// asking rather than always settling for the nearest slot.
    truecolor: bool,
    /// What is currently on the last three rows of the screen.
    ///
    /// A sweep prints two kinds of thing at once: a bar that stays put and
    /// findings that scroll past above it. That only works if every writer
    /// erases the bar, prints, and puts it back -- so the bar is held here, and
    /// `above` is the only way anything reaches the terminal. Leaving each
    /// caller to clear it themselves is what made a find blank the bar until the
    /// next redraw two seconds later.
    screen: Mutex<Screen>,
}

/// The bottom of the screen, which several threads write to at once.
#[derive(Default)]
struct Screen {
    /// The progress line, without its leading return or trailing erase. Empty
    /// when there is nothing drawn there.
    ///
    /// The bar is drawn with a blank row on either side, so it reads as a status
    /// line rather than as one more entry in the list of findings, and never sits
    /// flush against the bottom edge of the terminal. Those three rows are what
    /// every writer has to erase and restore, never just the bar's own -- and the
    /// cursor parks on the last of them, one below the bar, which is what each
    /// erase counts up from.
    sticky: String,
    /// Whether the last line of the transcript is blank.
    ///
    /// The transcript is everything that scrolls: the bar and the row above it
    /// are not part of it, so this stays true while the bar sits in the banner's
    /// trailing blank. It decides who provides the bar's gap row the first time
    /// it is drawn, and whether a block that wants clear air above it has to open
    /// any. Getting it wrong shows either as a doubled gap or as two blocks
    /// jammed together, depending on which way it is assumed.
    gap_ready: bool,
    /// Whether the bar's gap row is a blank the transcript printed rather than
    /// one the bar opened for itself.
    ///
    /// A borrowed row has to be given back: the erase that clears the bar takes
    /// both rows, and without this the separator between the banner and the
    /// findings would be swallowed by whichever find happened to print first.
    borrowed_gap: bool,
}

impl Screen {
    /// Decide where the first bar's gap row comes from, and remember which way it
    /// went. A transcript that already ends in a blank lends the bar that row;
    /// otherwise the bar opens one, which is what this returns.
    fn take_gap(&mut self) -> bool {
        self.borrowed_gap = self.gap_ready;
        !self.gap_ready
    }

    /// Give back a borrowed gap row, once. Called with the cursor sitting on that
    /// row having just erased it, so a true answer means the caller owes the
    /// transcript a blank line before it prints anything of its own.
    fn reclaim_gap(&mut self) -> bool {
        let borrowed = std::mem::take(&mut self.borrowed_gap);
        self.gap_ready |= borrowed;
        borrowed
    }
}

impl Ui {
    pub fn new(stream: Stream) -> Self {
        let allowed = std::env::var_os("NO_COLOR").is_none()
            && std::env::var("TERM").is_ok_and(|t| t != "dumb");
        let on_terminal = match stream {
            Stream::Stdout => std::io::stdout().is_terminal(),
            Stream::Stderr => std::io::stderr().is_terminal(),
        };
        let ui = Self {
            stream,
            color: allowed && on_terminal,
            interactive: allowed && std::io::stderr().is_terminal(),
            truecolor: std::env::var("COLORTERM")
                .is_ok_and(|v| v.contains("truecolor") || v.contains("24bit")),
            screen: Mutex::new(Screen::default()),
        };
        // The cursor spends a sweep parked at the end of a progress bar being
        // redrawn under it, which reads as flicker. Hidden for the length of the
        // run and restored by `Drop`.
        if ui.interactive {
            eprint!("{HIDE_CURSOR}");
            let _ = std::io::stderr().flush();
        }
        ui
    }

    /// Whether progress can be drawn in place. Callers use it to decide how often
    /// to report: every couple of seconds on a terminal, once a minute in a log.
    pub fn interactive(&self) -> bool {
        self.interactive
    }

    /// The escape that selects a palette colour at the depth this terminal has.
    fn sgr(&self, color: &Color) -> String {
        let (r, g, b) = color.rgb;
        if self.truecolor {
            format!("\x1b[38;2;{r};{g};{b}m")
        } else {
            format!("\x1b[38;5;{}m", color.index)
        }
    }

    fn paint(&self, code: &str, text: &str) -> String {
        if self.color {
            format!("{code}{text}{RESET}")
        } else {
            text.to_string()
        }
    }

    fn tint(&self, color: &Color, text: &str) -> String {
        self.paint(&self.sgr(color), text)
    }

    pub fn dim(&self, text: &str) -> String {
        self.paint(DIM, text)
    }

    /// The run's headline figures.
    pub fn headline(&self, text: &str) -> String {
        self.tint(&HEADLINE, text)
    }

    /// A filter hit, which is the whole point of a sweep.
    pub fn found(&self, text: &str) -> String {
        self.tint(&FOUND, text)
    }

    /// Addresses, hashes and phrases: long strings the eye has to pick out.
    pub fn data(&self, text: &str) -> String {
        self.tint(&DATA, text)
    }

    fn line(&self, text: &str) {
        match self.stream {
            Stream::Stdout => {
                println!("{text}");
                self.screen().gap_ready = text.is_empty();
            }
            Stream::Stderr => self.above(&[text.to_string()]),
        }
    }

    /// The bottom-of-screen state, recovered rather than propagated if a writer
    /// panicked holding it: a poisoned lock here would cost the run its output,
    /// and the worst a stale bar can do is need one more redraw.
    fn screen(&self) -> std::sync::MutexGuard<'_, Screen> {
        self.screen.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Print lines above the progress bar, leaving it drawn underneath.
    fn above(&self, lines: &[String]) {
        self.write_above(lines, false);
    }

    /// The same, for a block that is not part of the list it lands in the middle
    /// of: a blank line is opened first unless the transcript already ends in one.
    ///
    /// An interrupt is neither a finding nor a continuation of the finding above
    /// it, and reads as one when it inherits its spacing.
    fn above_spaced(&self, lines: &[String]) {
        self.write_above(lines, true);
    }

    /// The bar owns the bottom of the screen, so anything else has to erase it, print, and
    /// put it back in one go. Everything that writes to stderr goes through here,
    /// which is also what keeps two threads announcing at the same moment from
    /// interleaving halfway through each other's block: the lock is held for the
    /// whole write, not for each line.
    fn write_above(&self, lines: &[String], spaced: bool) {
        let mut screen = self.screen();
        let mut out = std::io::stderr().lock();
        if self.interactive {
            // Erase the row the cursor is parked on, then step up over the bar
            // and its gap row and erase those too, which leaves the cursor
            // exactly where the new lines belong. Erasing only the bar's own row
            // would push a blank line into the transcript on every find.
            let _ = write!(out, "{ERASE_ROW}");
            if !screen.sticky.is_empty() {
                let _ = write!(out, "{CURSOR_UP}{ERASE_ROW}{CURSOR_UP}{ERASE_ROW}");
                // Unless that row was the transcript's own blank, which is written
                // back before anything else: it is the separator under the banner,
                // and the bar was only borrowing it until something needed the room.
                if screen.reclaim_gap() {
                    let _ = writeln!(out);
                }
            }
        }
        if spaced && !screen.gap_ready {
            let _ = writeln!(out);
        }
        for line in lines {
            let _ = writeln!(out, "{line}");
        }
        if let Some(last) = lines.last() {
            screen.gap_ready = last.is_empty();
        }
        if self.interactive && !screen.sticky.is_empty() {
            // The first newline opens the gap row and the bar goes on the one
            // below it; the second opens the blank the cursor parks on.
            let _ = write!(out, "\n{ERASE_ROW}{}\n{ERASE_ROW}", screen.sticky);
        }
        let _ = out.flush();
    }

    /// Product line at the top of a run.
    pub fn title(&self, version: &str) {
        let name = if self.color {
            format!("{BOLD}{}keyforge{RESET}", self.sgr(&HEADLINE))
        } else {
            "keyforge".to_string()
        };
        self.line(&format!("\n  {name} {}", self.dim(version)));
    }

    /// A labelled line: the label right-aligned in the gutter, the value
    /// beginning at the shared value column.
    pub fn row(&self, label: &str, value: &str) {
        // Pad before colouring, or the escape bytes eat the field width.
        let padded = format!("{label:>GUTTER$}");
        self.line(&format!("  {}  {value}", self.dim(&padded)));
    }

    /// A row whose value is a headline figure.
    pub fn row_strong(&self, label: &str, value: &str) {
        self.row(label, &self.headline(value));
    }

    /// A further line belonging to the row above, aligned under its value.
    pub fn cont(&self, text: &str) {
        self.line(&format!("{}{}", " ".repeat(VALUE_COL), self.dim(text)));
    }

    /// A continuation line that is content rather than commentary, so it is not
    /// dimmed away: an address, a phrase, a hash.
    pub fn cont_plain(&self, text: &str) {
        self.line(&format!("{}{text}", " ".repeat(VALUE_COL)));
    }

    /// Blank separator between groups of rows.
    ///
    /// Consecutive gaps collapse into one, so a caller can ask for a separator
    /// without first working out whether whatever ran before it already left a
    /// blank line behind -- which for a sweep depends on whether it found
    /// anything, and is not knowable where the call is written.
    pub fn gap(&self) {
        if self.screen().gap_ready {
            return;
        }
        self.line("");
    }

    /// A labelled line for the stderr blocks, in the given colour when that
    /// stream takes one -- which is what `interactive` answers, the row stream's
    /// own `color` having nothing to say about where these lines go.
    fn labelled(&self, color: &Color, bold: bool, label: &str, text: &str) -> String {
        let padded = format!("{label:>GUTTER$}");
        let label = if self.interactive {
            let weight = if bold { BOLD } else { "" };
            format!("{weight}{}{padded}{RESET}", self.sgr(color))
        } else {
            padded
        };
        format!("  {label}  {text}")
    }

    /// A labelled line, plus any further lines of the message aligned under it.
    ///
    /// The longer diagnostics are written as several sentences across more than
    /// one line -- a refused resume explains what it found and then what to do
    /// about it -- and without this the second line starts at column 0 and the
    /// table falls apart exactly where someone is trying to read it.
    fn labelled_block(&self, color: &Color, bold: bool, label: &str, text: &str) -> Vec<String> {
        let mut rest = text.lines();
        let first = rest.next().unwrap_or_default();
        let mut lines = vec![self.labelled(color, bold, label, first)];
        lines.extend(rest.map(|line| format!("{}{line}", " ".repeat(VALUE_COL))));
        lines
    }

    /// Non-fatal problems, on stderr so they survive redirection and so the bar
    /// they interrupt is put back under them.
    pub fn warn(&self, text: &str) {
        self.above(&self.labelled_block(&WARNING, false, "warning", text));
    }

    pub fn error(&self, text: &str) {
        self.above(&self.labelled_block(&FAILURE, true, "error", text));
    }

    /// Something that is neither a result nor a fault: the run being interrupted.
    pub fn notice(&self, label: &str, text: &str) {
        self.above_spaced(&self.labelled_block(&NOTICE, true, label, text));
    }

    /// A link in a failure's cause chain, under the error it explains. On stderr
    /// with the error itself, rather than on the row stream, so a redirected run
    /// keeps the whole failure together.
    pub fn error_cause(&self, text: &str) {
        let lines: Vec<String> = text
            .lines()
            .map(|line| {
                let body = if self.interactive {
                    format!("{DIM}{line}{RESET}")
                } else {
                    line.to_string()
                };
                format!("{}{body}", " ".repeat(VALUE_COL))
            })
            .collect();
        self.above(&lines);
    }

    /// Announce a find the moment it happens: a labelled headline and however
    /// many aligned lines belong under it.
    ///
    /// Always on stderr, and always preceded by a clear, because on a terminal
    /// it is interrupting the progress bar it will be redrawn under. Written as
    /// one call rather than a row plus continuations so the clear is applied
    /// once, to the whole block.
    pub fn announce(&self, label: &str, headline: &str, body: &[String]) {
        let mut lines = vec![self.labelled(&FOUND, true, label, headline)];
        lines.extend(
            body.iter()
                .map(|line| format!("{}{line}", " ".repeat(VALUE_COL))),
        );
        self.above(&lines);
    }

    /// Report progress: a bar redrawn in place on a terminal, a plain appended
    /// line in a log.
    ///
    /// A sweep runs for days and is normally detached, where an in-place bar is
    /// actively harmful -- escape codes litter the file and a redraw every few
    /// seconds becomes hundreds of thousands of lines. So the two cases are
    /// genuinely different renderings, not one with the colour taken out.
    ///
    /// `unit` names what `done` and `total` count. The log line spells it out where
    /// the bar leaves it to the facts beside it, and it is not always seeds: a scan
    /// sweeping several `--offset` values counts walks of the range, not seeds in it.
    pub fn progress(&self, label: &str, done: u64, total: u64, unit: &str, facts: &[String]) {
        let percent = percent(done, total);

        if !self.interactive {
            self.above(&[format!(
                "  {label:>GUTTER$}  {percent:.1}%  {}/{} {unit} · {}",
                commas(done),
                commas(total),
                facts.join(" · ")
            )]);
            return;
        }

        const WIDTH: usize = 24;
        let filled = if total == 0 {
            WIDTH
        } else {
            (done.min(total) as usize * WIDTH) / total.max(1) as usize
        };
        // The track is dim rather than brand: only the filled part is the figure
        // worth reading, and colouring both makes the bar look full at a glance
        // whatever it says.
        let bar = format!(
            "{}{}{RESET}{DIM}{}{RESET}",
            self.sgr(&HEADLINE),
            "█".repeat(filled),
            "░".repeat(WIDTH.saturating_sub(filled))
        );
        // Drawn in the gutter layout too, so the bar sits where the row that
        // replaces it will appear. A label wider than the gutter would push the
        // bar out of that column, so it is cut rather than allowed to shift.
        let fitted: String = label.chars().take(GUTTER).collect();
        let line = format!(
            "  {DIM}{fitted:>GUTTER$}{RESET}  {bar} {percent:5.1}%  {DIM}{}{RESET}",
            facts.join(" · ")
        );

        // Recorded before it is drawn, so that whatever prints next puts back
        // this version of the bar rather than the one before it.
        let mut screen = self.screen();
        // The first bar has to open its own gap row unless the line above is
        // already blank -- the banner ends with one, but a find between the banner
        // and this first draw will have used it up. Later bars are redrawn in place
        // on the row they already own, leaving their gap row alone.
        let drawn = !screen.sticky.is_empty();
        let opening = !drawn && screen.take_gap();
        screen.sticky.clear();
        screen.sticky.push_str(&line);

        let mut out = std::io::stderr().lock();
        if opening {
            let _ = writeln!(out);
        }
        if drawn {
            // A redraw starts from the blank row under the bar, one below the row
            // being rewritten.
            let _ = write!(out, "{CURSOR_UP}");
        }
        // The trailing newline reopens that blank and leaves the cursor on it, so
        // the pair of moves cancels out and a redraw scrolls nothing.
        let _ = write!(out, "{ERASE_ROW}{line}\n{ERASE_ROW}");
        let _ = out.flush();
    }

    /// Give up the bottom of the screen: erase the progress line and stop redrawing it.
    ///
    /// Called once the work behind the bar is over, so that the closing summary
    /// lands where the bar was instead of pushing it down the screen forever.
    pub fn clear(&self) {
        let mut screen = self.screen();
        let drawn = !screen.sticky.is_empty();
        screen.sticky.clear();
        if !self.interactive {
            return;
        }
        let mut out = std::io::stderr().lock();
        let _ = write!(out, "{ERASE_ROW}");
        if drawn {
            // All three rows go -- the blank the cursor was parked on, the bar,
            // and the gap row above it -- so the closing summary starts where the
            // bar was rather than several lines below it.
            let _ = write!(out, "{CURSOR_UP}{ERASE_ROW}{CURSOR_UP}{ERASE_ROW}");
            // The cursor lands on that gap row, so the line above it is now the
            // last finding rather than a blank -- and a row the bar had borrowed
            // has just gone with it, so there is nothing left to give back. Only
            // said when a bar was actually erased: claiming it otherwise is what
            // put a second blank line under a banner that had nothing to separate.
            screen.gap_ready = false;
            screen.borrowed_gap = false;
        }
        let _ = out.flush();
    }
}

/// Give the cursor back on every way out of the program that runs destructors --
/// a clean finish, an early error return, or a panic that unwinds. A terminal
/// left without a cursor is a broken terminal, so this is not left to the
/// success path to do.
impl Drop for Ui {
    fn drop(&mut self) {
        self.restore_cursor();
    }
}

impl Ui {
    fn restore_cursor(&self) {
        if self.interactive {
            show_cursor();
        }
    }
}

/// Show the cursor again, for the one exit that runs no destructors.
///
/// `Drop` covers every ordinary way out, but a second Ctrl-C calls `process::exit`
/// straight from the signal handler, which runs none. That path has to say so
/// itself -- an impatient interrupt is otherwise precisely the case that leaves
/// the operator with an invisible cursor in their shell. A free function because
/// the handler outlives every borrow and so cannot hold the `Ui`.
pub fn show_cursor() {
    eprint!("{SHOW_CURSOR}");
    let _ = std::io::stderr().flush();
}

/// Progress as a percentage, floored to the tenth it will be printed at.
///
/// Floored rather than rounded because a sweep spends its last fraction of a
/// percent on millions of seeds -- minutes of work -- and rounding shows 100.0%
/// for all of it, beside an ETA still counting down. Flooring means 100.0% is
/// only ever printed when the range really is finished.
pub fn percent(done: u64, total: u64) -> f64 {
    if total == 0 {
        return 100.0;
    }
    (done.min(total) as f64 / total as f64 * 1000.0).floor() / 10.0
}

/// Thousands separators, so ten-digit seed counts stay readable.
/// The same, for a count that can exceed `u64`.
///
/// Search spaces are `u128` -- `java-random`'s is 2^48 and nothing stops a future one
/// being wider -- and a point count printed through a `u64` would be silently wrong for
/// exactly the vulnerabilities where the number matters most.
pub fn commas_u128(value: u128) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub fn commas(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() * 4 / 3);
    for (index, c) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A short human duration, at two significant units: `42s`, `12m 30s`, `3h 42m`,
/// `33d 04h`.
///
/// A sweep's projections run from seconds to weeks, and `802:42:26` is not a
/// number anyone can read at a glance.
pub fn duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "--".to_string();
    }
    // A tenth of a second matters for a filter load or a short bench and is noise
    // for anything longer, so precision is dropped as soon as the number grows.
    if seconds < 10.0 {
        return format!("{seconds:.1}s");
    }
    let s = seconds as u64;
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m {:02}s", s / 60, s % 60),
        3600..86_400 => format!("{}h {:02}m", s / 3600, (s % 3600) / 60),
        _ => format!("{}d {:02}h", s / 86_400, (s % 86_400) / 3600),
    }
}

/// A byte count in the unit that keeps it to a small number.
pub fn bytes(count: u64) -> String {
    const GB: f64 = 1e9;
    const MB: f64 = 1e6;
    let n = count as f64;
    if n >= GB {
        format!("{:.1} GB", n / GB)
    } else if n >= MB {
        format!("{:.1} MB", n / MB)
    } else {
        format!("{count} B")
    }
}

/// Lowercase hex, for hash160s.
pub fn hex(data: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(data.len() * 2);
    for byte in data {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The banner's trailing blank belongs to the transcript, and the bar only
    /// borrows it. Losing that row is what once left the first candidate of a
    /// sweep jammed against the last banner line it should have been separated from.
    #[test]
    fn a_borrowed_gap_row_is_given_back_once() {
        let mut screen = Screen {
            gap_ready: true,
            ..Screen::default()
        };
        assert!(
            !screen.take_gap(),
            "the bar should sit in the blank, not add one"
        );
        assert!(screen.reclaim_gap(), "the first find owes that blank back");
        assert!(
            screen.gap_ready,
            "which puts the transcript back on a blank line"
        );
        assert!(
            !screen.reclaim_gap(),
            "but only the first find, not every one"
        );
    }

    /// The other way round: nothing to borrow, so the bar opens its own row and
    /// owes nothing when it is erased. Giving one back here would insert a blank
    /// between two findings that belong together.
    #[test]
    fn a_bar_that_opened_its_own_gap_row_owes_nothing() {
        let mut screen = Screen::default();
        assert!(
            screen.take_gap(),
            "no blank above, so the bar has to open one"
        );
        assert!(!screen.reclaim_gap());
        assert!(!screen.gap_ready);
    }

    /// The last fraction of a percent of a sweep is still minutes of work, so
    /// 100.0% has to mean finished rather than nearly finished -- otherwise the
    /// bar reads full beside an ETA that is still counting down.
    #[test]
    fn a_percentage_only_reads_full_when_it_is() {
        assert_eq!(percent(0, 1_000), 0.0);
        assert_eq!(percent(500, 1_000), 50.0);
        // 99.9999%: one seed short of the 372M-seed bx-era window.
        assert_eq!(percent(371_999_999, 372_000_000), 99.9);
        assert_eq!(percent(372_000_000, 372_000_000), 100.0);
        // Nothing to do is done, and no count can exceed the whole.
        assert_eq!(percent(0, 0), 100.0);
        assert_eq!(percent(10, 5), 100.0);
    }

    #[test]
    fn commas_group_from_the_right() {
        assert_eq!(commas(0), "0");
        assert_eq!(commas(999), "999");
        assert_eq!(commas(1_000), "1,000");
        assert_eq!(commas(4_294_967_296), "4,294,967,296");
    }

    /// Each band has to hand over to the next without a gap or an overlap, and
    /// the projections a sweep prints span all four.
    #[test]
    fn durations_step_through_every_unit() {
        assert_eq!(duration(0.0), "0.0s");
        assert_eq!(duration(2.25), "2.2s");
        assert_eq!(duration(10.0), "10s");
        assert_eq!(duration(59.9), "59s");
        assert_eq!(duration(60.0), "1m 00s");
        assert_eq!(duration(3599.0), "59m 59s");
        assert_eq!(duration(3600.0), "1h 00m");
        assert_eq!(duration(86_399.0), "23h 59m");
        assert_eq!(duration(86_400.0), "1d 00h");
        // A full 2^32 sweep at 2,287 seeds/s, which is the projection the bench
        // subcommand exists to produce.
        assert_eq!(duration(4_294_967_296.0 / 2287.0), "21d 17h");
    }

    /// An unfinished rate produces a NaN ETA, which must render as "unknown"
    /// rather than as a number.
    #[test]
    fn a_non_finite_duration_is_not_rendered_as_a_number() {
        assert_eq!(duration(f64::NAN), "--");
        assert_eq!(duration(f64::INFINITY), "--");
        assert_eq!(duration(-1.0), "--");
    }

    #[test]
    fn byte_counts_pick_a_readable_unit() {
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(7_617_111_912), "7.6 GB");
    }
}
