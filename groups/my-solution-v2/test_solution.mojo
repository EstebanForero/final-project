from std.testing import assert_equal, assert_true, assert_false

from solution import (
    Bitboard,
    select_best_move,
    negamax,
    heuristic,
    eval_window,
    window_score,
)

# Mirror the constants from solution.mojo
comptime STRIDE: UInt64 = 8   # WIDTH + 1
comptime BIG_SCORE: Int = 1000000


# ---------------------------------------------------------------------------
# Helpers — place pieces directly without going through make_move
# ---------------------------------------------------------------------------

def place_my(mut b: Bitboard, col: Int, row: Int):
    b.my_pieces |= UInt64(1) << (UInt64(col) * STRIDE + UInt64(row))
    if UInt8(row + 1) > b.heights[col]:
        b.heights[col] = UInt8(row + 1)

def place_opp(mut b: Bitboard, col: Int, row: Int):
    b.opp_pieces |= UInt64(1) << (UInt64(col) * STRIDE + UInt64(row))
    if UInt8(row + 1) > b.heights[col]:
        b.heights[col] = UInt8(row + 1)


# ---------------------------------------------------------------------------
# window_score
# ---------------------------------------------------------------------------

def test_window_score_blocked() raises:
    assert_equal(window_score(2, 1), 0, msg="mixed window must score 0")

def test_window_score_my_three() raises:
    assert_equal(window_score(3, 0), 5, msg="3 mine should score 5")

def test_window_score_my_two() raises:
    assert_equal(window_score(2, 0), 2, msg="2 mine should score 2")

def test_window_score_opp_three() raises:
    assert_equal(window_score(0, 3), -4, msg="3 opp should score -4")

def test_window_score_opp_two() raises:
    assert_equal(window_score(0, 2), -1, msg="2 opp should score -1")

def test_window_score_empty() raises:
    assert_equal(window_score(0, 0), 0, msg="empty window should score 0")


# ---------------------------------------------------------------------------
# check_win / check_win_opp — all four directions
# ---------------------------------------------------------------------------

def test_check_win_horizontal() raises:
    var b = Bitboard()
    place_my(b, 0, 0); place_my(b, 1, 0); place_my(b, 2, 0); place_my(b, 3, 0)
    assert_true(b.check_win(), msg="horizontal win not detected")

def test_check_win_vertical() raises:
    var b = Bitboard()
    place_my(b, 0, 0); place_my(b, 0, 1); place_my(b, 0, 2); place_my(b, 0, 3)
    assert_true(b.check_win(), msg="vertical win not detected")

def test_check_win_diagonal_slash() raises:
    # / diagonal: col=0 row=0, col=1 row=1, col=2 row=2, col=3 row=3
    var b = Bitboard()
    place_my(b, 0, 0); place_my(b, 1, 1); place_my(b, 2, 2); place_my(b, 3, 3)
    assert_true(b.check_win(), msg="/ diagonal win not detected")

def test_check_win_diagonal_backslash() raises:
    # \ diagonal: col=0 row=3, col=1 row=2, col=2 row=1, col=3 row=0
    var b = Bitboard()
    place_my(b, 0, 3); place_my(b, 1, 2); place_my(b, 2, 1); place_my(b, 3, 0)
    assert_true(b.check_win(), msg="\\ diagonal win not detected")

def test_no_win_three_in_a_row() raises:
    var b = Bitboard()
    place_my(b, 0, 0); place_my(b, 1, 0); place_my(b, 2, 0)
    assert_false(b.check_win(), msg="3-in-a-row must not count as a win")

def test_check_win_opp_horizontal() raises:
    var b = Bitboard()
    place_opp(b, 0, 0); place_opp(b, 1, 0); place_opp(b, 2, 0); place_opp(b, 3, 0)
    assert_true(b.check_win_opp(), msg="opp horizontal win not detected")


# ---------------------------------------------------------------------------
# is_valid_move
# ---------------------------------------------------------------------------

def test_is_valid_move_empty_board() raises:
    var b = Bitboard()
    for col in range(7):
        assert_true(b.is_valid_move(col), msg="all cols must be valid on empty board")

def test_is_valid_move_full_column() raises:
    var b = Bitboard()
    for row in range(6):
        place_my(b, 0, row)
    assert_false(b.is_valid_move(0), msg="full column must not be a valid move")


# ---------------------------------------------------------------------------
# make_move / undo_move
# ---------------------------------------------------------------------------

def test_make_move_increments_height() raises:
    var b = Bitboard()
    assert_equal(Int(b.heights[3]), 0)
    b.make_move(3)
    assert_equal(Int(b.heights[3]), 1, msg="height must increase after make_move")

def test_undo_move_restores_board() raises:
    var b = Bitboard()
    var original_my  = b.my_pieces
    var original_opp = b.opp_pieces
    b.make_move(3)
    b.undo_move(3)
    assert_equal(b.my_pieces,  original_my,  msg="my_pieces not restored after undo")
    assert_equal(b.opp_pieces, original_opp, msg="opp_pieces not restored after undo")
    assert_equal(Int(b.heights[3]), 0,        msg="height not restored after undo")


# ---------------------------------------------------------------------------
# piece_at
# ---------------------------------------------------------------------------

def test_piece_at_my() raises:
    var b = Bitboard()
    place_my(b, 2, 0)
    assert_equal(b.piece_at(0, 2), 1, msg="piece_at should return 1 for my piece")

def test_piece_at_opp() raises:
    var b = Bitboard()
    place_opp(b, 2, 0)
    assert_equal(b.piece_at(0, 2), -1, msg="piece_at should return -1 for opp piece")

def test_piece_at_empty() raises:
    var b = Bitboard()
    assert_equal(b.piece_at(0, 0), 0, msg="piece_at should return 0 for empty cell")


# ---------------------------------------------------------------------------
# heuristic
# ---------------------------------------------------------------------------

def test_heuristic_empty_board() raises:
    var b = Bitboard()
    assert_equal(heuristic(b), 0, msg="heuristic of empty board must be 0")

def test_heuristic_positive_for_my_advantage() raises:
    var b = Bitboard()
    place_my(b, 3, 0); place_my(b, 4, 0)  # 2-in-a-row center, opponent empty
    assert_true(heuristic(b) > 0, msg="heuristic must be positive when I have more threats")

def test_heuristic_negative_for_opp_advantage() raises:
    var b = Bitboard()
    place_opp(b, 0, 0); place_opp(b, 1, 0); place_opp(b, 2, 0)  # opp has 3-in-a-row
    assert_true(heuristic(b) < 0, msg="heuristic must be negative when opp has more threats")


# ---------------------------------------------------------------------------
# select_best_move — full search
# ---------------------------------------------------------------------------

def test_plays_winning_move_horizontal() raises:
    # 3-in-a-row at cols 0-2, row 0 — col 3 completes the win
    var b = Bitboard()
    place_my(b, 0, 0); place_my(b, 1, 0); place_my(b, 2, 0)
    assert_equal(select_best_move(b), 3, msg="should play col 3 to win horizontally")

def test_blocks_opponent_horizontal() raises:
    # Opponent has 3-in-a-row at cols 0-2 — must block col 3
    var b = Bitboard()
    place_opp(b, 0, 0); place_opp(b, 1, 0); place_opp(b, 2, 0)
    assert_equal(select_best_move(b), 3, msg="should block col 3")

def test_plays_winning_move_vertical() raises:
    # 3 stacked in col 0 rows 0-2 — row 3 (col 0) wins
    var b = Bitboard()
    place_my(b, 0, 0); place_my(b, 0, 1); place_my(b, 0, 2)
    assert_equal(select_best_move(b), 0, msg="should play col 0 to win vertically")

def test_blocks_opponent_vertical() raises:
    # Opponent stacked 3 in col 6 — must block col 6
    var b = Bitboard()
    place_opp(b, 6, 0); place_opp(b, 6, 1); place_opp(b, 6, 2)
    assert_equal(select_best_move(b), 6, msg="should block col 6")


# ---------------------------------------------------------------------------
# main — run all tests
# ---------------------------------------------------------------------------

def main() raises:
    # window_score
    test_window_score_blocked();       print("test_window_score_blocked: OK")
    test_window_score_my_three();      print("test_window_score_my_three: OK")
    test_window_score_my_two();        print("test_window_score_my_two: OK")
    test_window_score_opp_three();     print("test_window_score_opp_three: OK")
    test_window_score_opp_two();       print("test_window_score_opp_two: OK")
    test_window_score_empty();         print("test_window_score_empty: OK")

    # check_win
    test_check_win_horizontal();       print("test_check_win_horizontal: OK")
    test_check_win_vertical();         print("test_check_win_vertical: OK")
    test_check_win_diagonal_slash();   print("test_check_win_diagonal_slash: OK")
    test_check_win_diagonal_backslash(); print("test_check_win_diagonal_backslash: OK")
    test_no_win_three_in_a_row();      print("test_no_win_three_in_a_row: OK")
    test_check_win_opp_horizontal();   print("test_check_win_opp_horizontal: OK")

    # is_valid_move
    test_is_valid_move_empty_board();  print("test_is_valid_move_empty_board: OK")
    test_is_valid_move_full_column();  print("test_is_valid_move_full_column: OK")

    # make_move / undo_move
    test_make_move_increments_height(); print("test_make_move_increments_height: OK")
    test_undo_move_restores_board();   print("test_undo_move_restores_board: OK")

    # piece_at
    test_piece_at_my();                print("test_piece_at_my: OK")
    test_piece_at_opp();               print("test_piece_at_opp: OK")
    test_piece_at_empty();             print("test_piece_at_empty: OK")

    # heuristic
    test_heuristic_empty_board();      print("test_heuristic_empty_board: OK")
    test_heuristic_positive_for_my_advantage(); print("test_heuristic_positive_for_my_advantage: OK")
    test_heuristic_negative_for_opp_advantage(); print("test_heuristic_negative_for_opp_advantage: OK")

    # select_best_move
    test_plays_winning_move_horizontal(); print("test_plays_winning_move_horizontal: OK")
    test_blocks_opponent_horizontal(); print("test_blocks_opponent_horizontal: OK")
    test_plays_winning_move_vertical(); print("test_plays_winning_move_vertical: OK")
    test_blocks_opponent_vertical();   print("test_blocks_opponent_vertical: OK")

    print("\nAll 26 tests passed!")
