---
layout: post
title:  "Optimal Kick Return Paths (NFL Big Data Bowl 2022)"
date:   2022-01-06 23:00:00 -0400
categories: blog
---

The following is not exactly a blog post - rather, it's my team's submission to the 2022 NFL Big Data Bowl. The league released three seasons of tracking data from special teams plays, and asked for creative projects using that data to gain insights for NFL teams. 

# Introduction 

The basic question we seek to answer is:

> _What path should a kick returner take?_

Or, in the mind of a data scientist, can we devise a mathematical framework which can tell us the optimal return path for a kick returner given the current distribution of defenders and blockers on the field? This 2022 NFL Big Data Bowl project is our attempt to do so.

The basic idea is as follows:
### Model 
We begin by regressing the tracking data of kick returns, as summarized by 100 features, onto the value of a kick return. Note that the _value_ of a yard line is the expected points of the next score if you start at that yard line, and so the value of a kick return is simply the value of the end yard line of the kick return. This regression yields a function $$\hat{f}$$ which, given the tracking data of a kick return frame, returns the expected value of that kick return frame.

### Predict
Given $$\hat{f}$$ and the tracking data of all players on the field during a given kick return frame, we compute the kick returner's optimal next location by maximizing $$\hat{f}$$ over a set of candidate next locations for the kick returner. With this, we can compute the optimal kick return location at any frame, and we can compute a complete optimal kick return path by repeating the procedure over sequential frames. 

### Evaluate players
We conclude our study by comparing actual kick return movements to our computed optimal movements, and by devising a metric to judge a kick returner's decision making.

# The Regression

We use a multiple linear regression model primarily because we want control over an *explicit* model of kick return decision-making with interpretable and actionable weights associated with each input feature. We could instead use a neural network to (potentially) improve predictive power, but neural nets lack interpretability, and so we would not be able to say with any certainty *why* our model suggested certain return paths. Of course, predictive power is important, so this is a good area for future work. 

## Response Column

Our response variable (dependent variable) is $$V$$, the "value" of the ending yard line of a kickoff return. We define the value of a yardline $$z$$, where $$z$$ means $$z$$ yards away from scoring a touchdown, as the expected number of points of the next score given that you start with a first down at yard line $$z$$. To find $$V_z$$ for each $$z$$, we simply take the average value of all points of the next score from every first down at yard line $$z$$ from 2010 to 2018. Then, we smooth these values. We performed this computation using `nflFastR` data in a separate project [1]. Below is a plot of $$V_z$$ as a function of $$z$$.

![](https://drive.google.com/uc?export=view&id=1tbbTe1ws0iG02XtUi2VPHZGHWa64nur8)

For the remiander of the project, we use the $$x$$ coordinate of the $$\{(x,y)\}$$ grid representing the football field, where $$x \in [0,120]$$ denotes the yard line, $$y\in [0,53.3]$$ denotes the latitude, and the origin $$(0,0)$$ denotes the bottom left corner of the field. Thus the value of a kick return ending at yardline $$x$$ is $$V_{110-x}$$.

## Feature: Kick Returner's Yardline

$$x_{kr}$$, the $$x$$ yardline of the kick return in the current frame, is a feature because the value of a kick return frame depends on how close the kick returner is to the endzone.

## Feature: $$j^{th}$$-Projected-Gap-Distance

For each $$j \in \{1,...,11\}$$, we find the $$y$$-distance to the $$j^{th}$$ closest defender when the kick returner is projected onto the average $$x$$ location of the defenders. Mathematically, letting $$kr$$ be the index of the kick returner, $$dj$$ be the index of the $$j^{th}$$ closest defender, and $$\theta = -(dir_{kr} - 90)$$, the feature is given by
$$\Delta y_{dj} = (y_{dj} - y_{kr}) + (\frac{1}{11}\sum_{j=1}^{11}x_{dj} - x_{kr})\cdot \tan(\theta).$$

Here is a picture which illustrates the feature.

![](https://drive.google.com/uc?export=view&id=1RReSGzmIbjJ8-tV6SWtmBq1s2NrUJv0i)


## Indicator Variable: $$j^{th}$$-Defender-Cannot-Catch-The-Kick-Returner 

Letting $$dj$$ denote the $$j^{th}$$ closest defender to the kick returner, we want an indicator variable

$$C_j = \unicode{x1D7D9}\{ dj \text{ cannot catch }  kr\}.$$

We define $$C_j$$ as 1 if and only if $$dj$$ cannot get to the intersection $$**$$ in the explanatory image below before $kr$, which is calculated using the locations, speed, and direction of the $$kr$$ and $$dj$$. Note that $$C_j$$ itself is not a feature, but will be used in a feature below.

![](https://drive.google.com/uc?export=view&id=1rJHev2T2Jn8FqJUkC3jwZKAPfvP_uiZv)

## Indicator Variable: $$j^{th}$$-Defender-Is-Blocked 

Letting $dj$ denote the $j^{th}$ closest defender to the kick returner, we want an indicator variable

$$B_j = \unicode{x1D7D9}\{ dj \text{ is blocked} \}.$$

We define $$B_j$$ as 1 if and only if a blocker can get to the intersection $*$ in the explanatory image below before $dj$, which is calculated using the locations, speed, and direction of the blockers and $$dj$$. $$B_j$$ itself is not a feature, but will be used in a feature below.

**Note**: we assume that each blocker is only able to block a single defender. Defenders and blockers are considered in order of their proximity to the kick returner. For instance, we first consider the closest defender; if a blocker can intercept them, that blocker is ineligible when we consider the second closest defender. Thus, the $j$th defender is blocked if and only if their is an *available* defender that can reach intersection $*$ first. 

![](https://drive.google.com/uc?export=view&id=1o35rS5CHW0Cpk3pu83F88sJDvpfUcA29)

## Feature: Segmented Distance to the $$j^{th}$$ Closest Defender

First, let

$$A_j = (1-B_j) \cdot (1 - C_j)$$

be the $$j^{th}$$-_Able-to-Tackle_ indicator variable, which is 1 if and only if the $j^{th}$ defender can catch $kr$ and is not blocked. 

Denote the distance between the kick returner $$kr$$ and the $$j^{th}$$ closest defender $$dj$$ as
$$d(kr,dj) := \sqrt{(x_{kr}-x_{dj})^2 + (y_{kr}-y_{dj})^2}.$$

Let $$\lambda_1$$ and $$\lambda_2$$ be hyperparameters encoding how close $$kr$$ and $$dj$$ are in terms of $$x$$-distance in yards. For now, we set $$\lambda_1 = 2$$ and $$\lambda_2 = 2$$. If inclined, we can do a cross validation to determine better values for these hyperparameters.

Then, in our regression, we want terms indicating the distance of $$kr$$ to $$dj$$, with different coefficients depending on whether the defender is in front of or behind the kicker, whether the defender is blocked, and whether the defender can catch the kick returner. For example, a defender who is 5 yards away from the kick returner should be treated differently depending on whether he is blocked or not. Also, a defender who is 1 yard away from the kick returner should be treated differently depending on whether he is in front of or behind the returner. Defenders can have a vastly different impact on the outcome of the play depending on these conditions.

So, for the $$j^{th}$$ defender, we have 8 _segmented-distance_ features:
$$
\begin{align*}
  & d(kr,dj)  \cdot \unicode{x1D7D9}\{ x_{kr} - x_{dj} \in (\lambda_1,\infty) \} \cdot A_j \\
  & d(kr,dj) \cdot \unicode{x1D7D9}\{ x_{kr} - x_{dj} \in (\lambda_1,\infty) \} \cdot (1-A_j)  \\
  & d(kr,dj)  \cdot\unicode{x1D7D9}\{ x_{kr} - x_{dj} \in (0,\lambda_1] \} \cdot A_j \\
  & d(kr,dj)  \cdot\unicode{x1D7D9}\{ x_{kr} - x_{dj} \in (0,\lambda_1] \} \cdot (1-A_j) \\
  & d(kr,dj) \cdot \unicode{x1D7D9}\{ x_{kr} - x_{dj} \in (-\lambda_2,0] \} \cdot A_j  \\
  & d(kr,dj)  \cdot \unicode{x1D7D9}\{ x_{kr} - x_{dj} \in (-\lambda_2,0] \} \cdot (1-A_j) \\
  & d(kr,dj) \cdot \unicode{x1D7D9}\{ x_{kr} - x_{dj} \in (-\infty, -\lambda_2] \} \cdot A_j  \\
  & d(kr,dj) \cdot \unicode{x1D7D9}\{ x_{kr} - x_{dj} \in (-\infty, -\lambda_2] \} \cdot (1-A_j)  \\
\end{align*}
$$

## Run the Regression

Using these 100 features (1 $x_{kr}$ feature, 11 projected-gap-distance features, and 88 segmented distance features), and our response column $V$, we run an ordinary linear regression to obtain a function $\hat{f}$ which, given the tracking data of all players on the field during a kick return, returns the expected value of the play.

We train the regression on all kick returns (removing touchbacks) from 2018 and 2019, leaving all kick returns from 2020 as testing data. The out-of-sample rmse is $0.51$, which is fine considering the magnitude of $V$ and how noisy kick returns are.

# Algorithm: Choosing the Optimal Run Path for a Kick Returner

The result of our regression is a function $\hat f$ which, given the $(x,y,s,dir)$ variables of each player on the field during a given kick return frame, outputs the expected value of this kick return. 

To choose the optimal run path for a kick returner, we use a greedy algorithm. The basic idea is that, given the tracking data of all players on the field during the current frame of a kick return, we choose the kick returner's location for the next frame as the location which maximizes the value of $\hat{f}$, keeping the variables for all the other players the same. 

More specifically, suppose the kick returner's state at the current frame is given by $(x_{0,kr},y_{0,kr},s_{0,kr},a_{0,kr},dir_{0,kr})$. Let $\mathcal{C}$ denote the set of candidate kick returner states for the next frame. $\mathcal{C}$ consists of 16 equidistant points on the circle centered at $(x_{0,kr},y_{0,kr})$, having radius $s_{0,kr} \cdot \Delta t$, where $\Delta t = 0.1$ seconds is the time difference between frames. Moreover, if $(x_{1,kr},y_{1,kr})$ is a kick returner's candidate location for the next frame, then his candidate next direction $dir_{1,kr}$ is given by the angle from $(x_{0,kr},y_{0,kr})$ to $(x_{1,kr},y_{1,kr})$, and his candidate next speed is given by $s_{1,kr} = s_{0,kr} + a_{0,kr}\cdot \Delta t.$ Note that his candidate next acceleration $a_{1,kr}$ is irrelevant because acceleration is not used in our features or in $\hat f$. Moreover, we can't even compute $a_{1,kr}$ because we have no information about the kick returner's jerk (derivative of his acceleration). In symbols, a kick returner's optimal next state is given by

$$(x_{*,kr},y_{*,kr}) := \text{arg}\max_{\mathcal{C}} \hat{f}(x_{1,kr},y_{1,kr},s_{1,kr},dir_{1,kr}).$$

Using our optimization framework, we can construct the kick returner's optimal return path during a kick return. Our predictions can allow us to: 
1. Visualize and assess a returner's decision making on individual plays by comparing their observed path to a simulated optimal path.
2. Establish quantitative metrics for evaluating a returner's decision-making.

We provide examples and discuss in more detail below.

<!-- In fact, by substituting $kr$ for the index $j$ of any other player on the field in the above algorithm, we can compute the optimal movement for _any player_ on the field during a kick return. -->

# Examples: Compare a kick returner's actual movement to his computed optimal movement over the course of several frames. 

Here, we consider several example kick returns (all real plays from 2020 - our holdout dataset). 

At a given point in each of these kick returns, we start with the actual state of every player on the field, and compute the kick returner's optimal movement 5 frames ahead. We show the kick returner's <span style="color:green">computed optimal trajectory in green</span>, and show his <span style="color:red">observed trajectory in red</span>. We also show the observed trajectories of the blockers in black and the <span style="color:red">defenders in blue</span>.

### Example 1

![](https://drive.google.com/uc?export=view&id=1dJ_RN0bTHUWxdprIDGQlU6BpbtPmGQvx)

At the start of this kick return, the kick returner moves southwest, but our algorithm recommends moving northwest. Moving northwest involves going towards the open field, whereas going southwest involves moving towards the blockers.

### Example 2
![](https://drive.google.com/uc?export=view&id=1ttJNZAXt5doR_bN1WXYZpiW0mIu0ht7K)

At a critical juncture in this kick return, the kick returner moves backwards to try to run around the entire field, but ends up losing 15 yards. Our algorithm recommends moving slightly forwards and getting tackled.

### Example 3
![](https://drive.google.com/uc?export=view&id=1coGBI-9Sh9KIM_ohlv-pQ7kUucpY2wue)

The kick returner moves southwest towards his blockers, but our algorithm recommends moving northwest towards the open field.

# Example: For each frame, compare a kick returner's actual movement to his computed optimal movement.

Here, we consider an example kick return. At each frame, given the actual state of every player on the field, we compute the kick returner's optimal movement for the next frame. After each frame, we reset the location of each player to his actual location. The animation below shows the kick returner's actual movement (in red) and the optimal movement at each step (in green), along with the blockers (in black) and the defenders (in blue).

![](https://drive.google.com/uc?export=view&id=1pjxxuEakx8mulxW4yHq3RSlmWvbgvbTg)

At the beginning of the return, our algorithm wants the returner to move upwards towards the open field. As the play develops and that option is closed off by defenders, the algorithm changes tactics to try and find the best path that seeks open space and utilizes blockers.

This visualization - showing the frame-by-frame deviations of the algorithm from the returner's actual path - should help provide some intuition for the metrics we describe in the following section.

# Two metrics for evaluating a kick returner's decision making: _ADOM_ and _AEBO_

## _ADOM_

For the $$i^{th}$$ frame, let $$(x_{i+1,1},y_{i+1,1})$$ be the kick returner's observed next location, and let $$(x_{i+1,*},y_{i+1,*})$$ be our computed optimal next location for the kick returner. The distance $$d_i$$ between these 2 points is

$$d_i = \sqrt{(x_{i1}-x_{i*})^2 + (y_{i1}-y_{i*})^2}.$$

We define a kick returner's _average deviation from optimal movement_, or _ADOM_, as the distance between the observed and optimal next kick-returner locations averaged over all his kick-return frames,

$$ADOM := \frac{1}{n} \sum_{\text{frames } i=1}^{n} d_i. $$

A _lower_ $$ADOM$$ indicates better kick-return decision making.

For instance, the $$ADOM$$ of the example kick return shown above is $$0.55$$, indicating that the kick returner is on average about a half yard away from optimal movement during that kick return.

As we used all kick returns from 2018 and 2019 as training data for our regression, this leaves all 2020 kick returns as hold-out testing data. 

Rather than evaluate every frame of every play, to increase our computational efficiency we found the optimal next kick-returner location for every $$10^{th}$$ frame of every kick return from 2020. We then computed $$ADOM$$ for every kick returner in this period, which produced the rankings of the 10 best kick return decision makers, for kick returners recording at least 15 kick returns. Note that a lower _ADOM_ corresponds to better decision making during returns.

![](https://drive.google.com/uc?export=view&id=1_gVAdZ4l3wCvEdPqzApwELf2FIbM9lWV)

To understand how _ADOM_ relates to a more familiar metric used to judge kick returns, average return yardage, we plot _ADOM_ vs. avg. return yardage below.

![](https://drive.google.com/uc?export=view&id=1CKkXoh8kWIdFoK0ZWnX9Q3kzk_htEqrw)

There is no clear relationship across returners between ADOM and Average Return Yardage. This is not necessarily surprising; ADOM is not directly an outcome-based metric, meaning that it only reflects how close the returner's decisions are to the optimal decisions - ADOM does not tell us anything about the yardage that those decisions earn. So, a lower ADOM does not necessarily directly correspond to higher return yardage. ADOM weighs all decisions equally, so it reflects which players make good decisions **consistently** - not which players make critical good decisions resulting in high yardage gains. Inconsistent players and so-called "home run hitters" who take high-variance strategies might have high yardage numbers but a low ADOM.

To evaluate which players make **critical, high-impact** decisions that result in large yardage gains, we develop another metric.

## *AEBO*

The _expected value added_, or _EVA_, of moving from $$(x_{i0},y_{i0})$$ to the observed next location $$(x_{i+1,1},y_{i+1,1})$$ in the $$i^{th}$$ frame is 

$$EVA_{i,obs} = \hat f( x_{i+1,1}, y_{i+1,1}) - \hat f(x_{i0},y_{i0}).$$

Similarly, the _EVA_ of moving from $$(x_{i0},y_{i0})$$ to the computed optimal next location $$(x_{i+1,*},y_{i+1,*})$$ in the $i^{th}$ frame is 

$$EVA_{i,*} = \hat f( x_{i+1,*}, y_{i+1,*}) - \hat f(x_{i0},y_{i0}).$$

We define a kick returner's _average EVA below optimal_, or _AEBO_, as the difference in optimal EVA and observed EVA averaged over all his kick-return frames,

$$AEBO := \frac{1}{n} \sum_{\text{frames } i=1}^{n} \big( EVA_{i,*} - EVA_{i,obs} \big).$$

We computed $AEBO$ on the same dataset on which we computed _ADOM_, producing another ranking of the 10 best kick return decision makers, for kick returners recording at least 15 kick returns.

![](https://drive.google.com/uc?export=view&id=1unLj2w3u-IeYeEKHygQakSNBxLyweMuC)

A _lower_ $AEBO$ indicates better **critical** kick-return decision making. AEBO accomplishes this because it evaluates the *difference in predicted outcomes* based on decisions, rather than the difference in the decisions themselves. Therefore, it does not value all decisions equally - decisions are effectively weighted by their importance (as measured by EVA). 

As before, we plot _AEBO_ vs. avg. return yardage to gain insight on our metric.

![](https://drive.google.com/uc?export=view&id=1FvPVxE2RuFD3aCNZL8mcABZ9z3JzcMV5)

The correlation between AEBO and Average Return Yardage is striking - clearly players with a lower AEBO (and therefore better critical decision making) earn more yards.

This result, which comes from held out data (not the data our model was trained on), provides evidence that our model and our approach are valid. Our approach led to a metric _AEBO_, which gave us a way to evaluate kick returners and produced a ranking that matched our intuition of what it means to be a good kick returner - highly ranked players have high average  return yardage. Hence we feel validated in our approach. 

Moreover, _AEBO_ is blind to the _final_ outcome of a kick return, yet players with a low _AEBO_ are the players with a high avg. return yardage! This is a fantastic insight because it gives us a way of finding **hidden value**. NFL teams want players who produce a high avg. return yardage. Unfortunately, the final outcome of a kick return is very noisy, because players can benefit from an exceptional block or a lucky missed tackle, for instace. _AEBO_ mitigates this noise by aggregating over _all_ of a kick returner's frames, not just his final outcomes. 

NFL teams should consider using _AEBO_ as a tool to evaluate a kick returner's value. Players with a low _AEBO_ might be undervalued - they may be due for an increase in yardage sooner rather than later. On the other hand, players with a high _AEBO_ may be overrated - perhaps they're just getting lucky and are due for a regression when their poor decision-making catches up with them.

# Conclusion

We devised a novel framework for computing the optimal return path for a kick returner. Our pipeline of regressing kick return tracking data on the value of a kick return gave us a function $\hat f$ to compute the expected value of a kick return frame, and we maximized this $\hat{f}$ to find the next location for a kick returner to move to. 

* We used our algorithm to evaluate kick returner's decision making and to examine critical junctures in kick returns. This is potentially valuable for NFL coaches to show players where they could have made better decisions. 
* Our _AEBO_ metric allows us to quantitatively evaluate a kick returner's gameplay without being overly reliant on the noisy final outcomes of his kick returns. This is potentially valuable for NFL teams looking to evaluate personnel and find hidden talent. 

## Future Work

There is still much room for improvement, which is expected when working on such a complex problem. Future work includes:

* Improve the features and consider feature importance.
* Consider a deep neural network instead of a regression for $\hat{f}$.
* Consider using a minimax-game-tree framework, rather than a greedy algorithm, for computing the optimal return path.

## Our Code

We wrote all our code in a single `Colab` file, which can be found at the following link.

> https://drive.google.com/file/d/1e9kIU5ZTCLfQdKfZP3s1mqFVT_wX-7Gc/view?usp=sharing

## Citations

1. Value_1st_Down: `1stdown_model1.R` from https://github.com/snoopryan123/x_pts_nfl/tree/main/code2

## Find Us on Twitter

Ryan Brill: `@RyanBrill_`

Joey Rudoler: `@JRudoler`

Tai Nguyen: `@taidn97`
