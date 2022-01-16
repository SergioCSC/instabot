from pandas import DataFrame
import datetime
from dataclasses import dataclass, field
from collections import Counter, Iterable, Iterator, Sized

POSTS_N = 'p'
LIKES_N = 'l'
COMMENTS_N = 'c'

# def add_to_counter(df: DataFrame, index_: str, posts_count_: int,
#                    likes_count_: int, comments_count_: int) -> None:
#     df.loc[index_] = df.loc[index_] + (posts_count_, likes_count_, comments_count_)

# class PLCs(Counter):
#     posts_count: float
#     likes_count: float
#     comments_count: float
#


@dataclass
class UserPostsInfo:
    overall: Counter = field(default_factory=Counter)

    m_x0_to_x5_r: Counter = field(default_factory=Counter)

    m_00_to_10_r: Counter = field(default_factory=Counter)
    m_10_to_20_r: Counter = field(default_factory=Counter)
    m_20_to_30_r: Counter = field(default_factory=Counter)
    m_30_to_40_r: Counter = field(default_factory=Counter)
    m_40_to_50_r: Counter = field(default_factory=Counter)
    m_50_to_60_r: Counter = field(default_factory=Counter)

    m_var: Counter = field(default_factory=Counter)

    h_3_to_9_r: Counter = field(default_factory=Counter)
    h_9_to_15_r: Counter = field(default_factory=Counter)
    h_15_to_21_r: Counter = field(default_factory=Counter)
    h_21_to_3_r: Counter = field(default_factory=Counter)

    h_var: Counter = field(default_factory=Counter)

    weekend_r: Counter = field(default_factory=Counter)

    d_01_to_10_r: Counter = field(default_factory=Counter)
    d_11_to_20_r: Counter = field(default_factory=Counter)
    d_21_to_31_r: Counter = field(default_factory=Counter)

    d_var: Counter = field(default_factory=Counter)

    winter_r: Counter = field(default_factory=Counter)
    spring_r: Counter = field(default_factory=Counter)
    summer_r: Counter = field(default_factory=Counter)
    autumn_r: Counter = field(default_factory=Counter)

    season_var: Counter = field(default_factory=Counter)


def make_user_posts_dataframe():
    columns = (POSTS_N, LIKES_N, COMMENTS_N)
    indexes = tuple(UserPostsInfo.__annotations__)
    user_posts_info = DataFrame(0.0, index=indexes, columns=columns)
    return user_posts_info


def add_to_counter(counter_: Counter, posts_count_: int, likes_count_: int,
                   comments_count_: int) -> None:
    counter_.update({POSTS_N: posts_count_,
                     LIKES_N: likes_count_,
                     COMMENTS_N: comments_count_})


def pick_by_minutes(upi_: UserPostsInfo, minutes_: int,
                    posts_count_: int, likes_count_: int, comments_count_: int):

    if minutes_ < 0 or minutes_ > 59:
        raise ValueError(f'error minutes: {minutes_}')

    if minutes_ % 10 < 5:
        add_to_counter(upi_.m_x0_to_x5_r, 1, likes_count_, comments_count_)

    m10 = minutes_ // 10
    add_to_counter(upi_.__getattribute__(f'm_{m10}0_to_{m10 + 1}0_r'), 1,
                   likes_count_, comments_count_)
    # if minutes_ < 10:
    #     add_to_counter(upi_.m_00_to_10_r, 1, likes_count_, comments_count_)
    # elif 10 <= minutes_ < 20:
    #     add_to_counter(upi_.m_10_to_20_r, 1, likes_count_, comments_count_)
    # elif 20 <= minutes_ < 30:
    #     add_to_counter(upi_.m_20_to_30_r, 1, likes_count_, comments_count_)
    # elif 30 <= minutes_ < 40:
    #     add_to_counter(upi_.m_30_to_40_r, 1, likes_count_, comments_count_)
    # elif 40 <= minutes_ < 50:
    #     add_to_counter(upi_.m_40_to_50_r, 1, likes_count_, comments_count_)
    # elif 50 <= minutes_:
    #     add_to_counter(upi_.m_50_to_60_r, 1, likes_count_, comments_count_)
    # else:
    #     raise ValueError(f'error minutes: {minutes_}')


def pick_by_hours(upi_: UserPostsInfo, hours_: int,
                  posts_count_: int, likes_count_: int, comments_count_: int):
    if hours_ < 0 or hours_ > 23:
        raise ValueError(f'error hours: {hours_}')
    if hours_ < 3 or hours_ >= 21:
        add_to_counter(upi_.h_21_to_3_r, 1, likes_count_, comments_count_)
    elif 3 <= hours_ < 9:
        add_to_counter(upi_.h_3_to_9_r, 1, likes_count_, comments_count_)
    elif 9 <= hours_ < 15:
        add_to_counter(upi_.h_9_to_15_r, 1, likes_count_, comments_count_)
    elif 15 <= hours_ < 21:
        add_to_counter(upi_.h_15_to_21_r, 1, likes_count_, comments_count_)


def pick_by_date(upi_: UserPostsInfo, date_: str,
                 posts_count_: int, likes_count_: int, comments_count_: int):

    day, month, year = (int(t) for t in date_.split('.'))
    if day < 1 or day > 31:
        raise ValueError(f'error day: {day} date: {date_}')
    if month < 1 or month > 12:
        raise ValueError(f'error month: {month} date: {date_}')
    if year < 2010 or year > (datetime.datetime.now() + datetime.timedelta(1)).year:
        raise ValueError(f'error year: {year} date: {date_}')

    day_of_week = datetime.date(year, month, day).weekday()
    if day_of_week >= 5:  # 5 is Saturday, 6 is Sunday
        add_to_counter(upi_.weekend_r, 1, likes_count_, comments_count_)

    if day <= 10:
        add_to_counter(upi_.d_01_to_10_r, 1, likes_count_, comments_count_)
    elif 10 < day <= 20:
        add_to_counter(upi_.d_11_to_20_r, 1, likes_count_, comments_count_)
    elif 20 < day:
        add_to_counter(upi_.d_21_to_31_r, 1, likes_count_, comments_count_)

    if month < 3 or month == 12:
        add_to_counter(upi_.winter_r, 1, likes_count_, comments_count_)
    elif 3 <= month < 6:
        add_to_counter(upi_.spring_r, 1, likes_count_, comments_count_)
    elif 6 <= month < 9:
        add_to_counter(upi_.summer_r, 1, likes_count_, comments_count_)
    elif 9 <= month < 12:
        add_to_counter(upi_.autumn_r, 1, likes_count_, comments_count_)
    else:
        raise ValueError(f'error month: {month} date: {date_}')


def variance_dict(*counters) -> dict:
    def bessel_variance(l: Sized) -> float:  # same as statistics.variance()
        mean = sum(l) / len(l)
        return sum((x - mean) ** 2 for x in l) / (len(l) - 1)

    posts_variance = bessel_variance([c[POSTS_N] for c in counters])
    likes_variance = bessel_variance([c[LIKES_N] for c in counters])
    comments_variance = bessel_variance([c[COMMENTS_N] for c in counters])

    return {POSTS_N: posts_variance, LIKES_N: likes_variance, COMMENTS_N: comments_variance}



