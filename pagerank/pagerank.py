import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_dist = {}
    if len(corpus[page]) == 0:
        for unlinked_pages in corpus:
            prob_dist[unlinked_pages] = 1/len(corpus)
    else:
        for linked_pages in corpus[page]:
            prob_dist[linked_pages] = damping_factor/(len(corpus[page]))+(1-damping_factor)/len(corpus)
        for unlinked_pages in (corpus.keys()-corpus[page]):
            prob_dist[unlinked_pages] = (1-damping_factor)/len(corpus)

    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    appearance = {}
    sample = 0
    for pages in corpus:
        appearance[pages] = 0
    for i in range(n):
        if sample == 0:
            sample = random.choice(list(corpus.keys()))
            appearance[sample] += 1
        else:
            new_probdist = transition_model(corpus, sample, damping_factor)
            sample = random.choices(list(new_probdist.keys()), weights=list(new_probdist.values()))[0]
            appearance[sample] += 1
    pagerank = {}
    for pages in list(appearance.keys()):
        pagerank[pages]=(appearance[pages]/n)
    print(f'Pagerank Sample Sum: {round(sum(pagerank.values()),5)}')
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = {}
    new_pagerank = {}
    for pages in corpus:
        pagerank[pages] = 1/len(corpus)
    count = 1
    while count > 0.001:
        count = 0
        for pages in pagerank:
            links = 0
            for pages_linked in pagerank:
                if pages in corpus[pages_linked]:
                    links += pagerank[pages_linked]/len(corpus[pages_linked])
            new_pagerank[pages] = (1-damping_factor)/len(corpus) + damping_factor*links
            if count < (abs(new_pagerank[pages] - pagerank[pages])):
                count = abs(new_pagerank[pages] - pagerank[pages])
            pagerank[pages] = new_pagerank[pages]
    totalpagerank = sum(pagerank.values())
    for pages in pagerank:
        pagerank[pages] = pagerank[pages]/totalpagerank
    print(f'Pagerank Iteration Sum: {round(sum(pagerank.values()),5)}')
    return pagerank

if __name__ == "__main__":
    main()
