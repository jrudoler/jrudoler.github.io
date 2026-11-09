---
layout: default
title: About
permalink: /
description: "Joey Rudoler. Statistics, deep learning, and neuroscience at Wharton."
---
<main id="main" class="home" tabindex="-1">
<a class="tetons-cover" href="{{ '/images/tetons_wall.jpg' | relative_url }}" aria-label="View the full photograph of Joey hiking in the Tetons">{% include responsive-image.html name="tetons" alt="Joey hiking beside a stream and rock face in the Tetons" sizes="(max-width: 640px) calc(100vw - 40px), (max-width: 914px) calc(100vw - 64px), 850px" loading="eager" priority="high" %}</a>
<header class="bio-heading"><h1>Joey Rudoler</h1></header>
<section class="bio" aria-label="About me" markdown="1">
<figure class="bio-photo">{% include responsive-image.html name="headshot" alt="Joey Rudoler" sizes="(max-width: 640px) 115px, 165px" loading="eager" %}</figure>

I'm a PhD student at The Wharton School, where I study Statistics and Data Science. My interests span deep learning theory, neuroscience, and the statistics of both.

I was drawn to statistics and machine learning by a desire to find my place in a world overwhelmed with data. There are lots of problems I'd like to tackle (probably too many), and you can read about some of my past work and current interests on the [research page](/research/). I love thinking about the world both experientially and mathematically, and I honestly believe we need both perspectives to have any idea what's going on around us.

<div class="bio-actions"><a href="#research">{% include icons/arrow-down.svg %} Explore my work</a><a href="{{ '/files/RudolerCV.pdf' | relative_url }}">{% include icons/file-text.svg %} CV <span class="file-label">PDF</span></a><a href="mailto:jrudoler@wharton.upenn.edu">{% include icons/mail.svg %} Get in touch</a></div>

<details class="background"><summary>Background</summary><div markdown="1">
Before starting my PhD, I worked as a software engineer facilitating computational cognitive neuroscience research at Penn’s [Computational Memory Lab](http://memory.psych.upenn.edu/Main_Page). While working for the lab I finished a part-time engineering master's in Data Science, which included a thesis supervised by [Konrad Kording](https://kordinglab.com/). I attended Penn for undergrad as well, majoring in Physics & Astronomy with a Concentration in Business and Technology, along with minors in English and Music.

</div></details>
</section>

<section id="research" class="home-section" aria-labelledby="research-title">
<h2 id="research-title">Research</h2>
<p>My research has wandered from machine learning applications in neuroscience toward the theory and methodology of deep learning. Across both areas, I'm interested in how learning systems pick up useful patterns, and when those patterns carry over to new people, data, or tasks.</p>
<p class="section-links"><a href="{{ '/research/' | relative_url }}">Research notes & figures</a> · <a href="{{ '/publications/' | relative_url }}">All publications</a></p>
<h3>Selected work</h3>
<div class="selected-work">
{% assign papers = site.data.publications | sort: 'date' | reverse %}
{% for paper in papers %}
{% assign details = site.data.publication_details[paper.key] %}
{% if details.selected %}
<article class="work-entry"><span class="work-year">{{ paper.year }}</span><div><h4><a href="{{ '/publications/' | relative_url }}#{{ paper.key }}">{{ paper.title | escape }}</a></h4><p class="work-venue">{{ paper.venue | default: paper.type | escape }}</p><details class="work-details"><summary>Authors & links</summary><p>{% for author in paper.authors %}{{ author | escape }}{% unless forloop.last %}, {% endunless %}{% endfor %}</p><div class="work-links">{% if paper.url != '' %}<a href="{{ paper.url | escape }}">{% include icons/external-link.svg %} Paper</a>{% endif %}{% if paper.preprint != '' %}<a href="{{ paper.preprint | escape }}">Preprint</a>{% endif %}{% for link in details.links %}<a href="{{ link.url | relative_url }}">{{ link.label }}</a>{% endfor %}<a href="{{ '/publications/' | relative_url }}#{{ paper.key }}">{% include icons/quote.svg %} Citation</a></div></details></div></article>
{% endif %}
{% endfor %}
</div>
</section>
<section class="home-section" aria-label="Recent announcements">{% include news-feed.html title="Recent news" limit=3 show_summary=false show_more_link=true more_url='/news/' date_format="%b %Y" %}</section>
<aside class="personal-note">
{% include epigraph.html text="In works of labour, or of skill,<br>I would be busy too;<br>For Satan finds some mischief still,<br>For idle hands to do." author="Isaac Watts" %}
</aside>
</main>
