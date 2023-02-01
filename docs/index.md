# YData Synthetic


## TEST 

$$
\operatorname{ker} f=\{g\in G:f(g)=e_{H}\}{\mbox{.}}
$$

The `#!python range()` function is used to generate a sequence of numbers.
``` py title="bubble_sort.py" linenums="1" hl_lines="2 3"
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j] # (1)
```

1. :A lot of loops!

``` yaml
theme:
  features:
    - content.code.annotate # (1)
```

1. :man_raising_hand: I'm a code annotation! I can contain `code`, __formatted
    text__, images, ... basically anything that can be expressed in Markdown[^1].

[^1]: Lorem ipsum dolor sit amet, consectetur adipiscing elit.

## Installation

=== "Windows"

    For Windows

=== "Linux"

    For Linux


| Method      | Description                          |
| ----------- | ------------------------------------ |
| `GET`       | :material-check:     Fetch resource  |
| `PUT`       | :material-check-all: Update resource |
| `DELETE`    | :material-close:     Delete resource |

## Test

``` mermaid
sequenceDiagram
  Alice->>John: Hello John, how are you?
  loop Healthcheck
      John->>John: Fight against hypochondria
  end
  Note right of John: Rational thoughts!
  John-->>Alice: Great!
  John->>Bob: How about you?
  Bob-->>John: Jolly good!
```



<figure markdown>
  ![Image title](https://dummyimage.com/600x400/){ width="300" loading=lazy }
  <figcaption>Image caption</figcaption>
</figure>

## Roadmap

- [X] Important thing 1
    - [ ] Define stuff
- [X] Important thing 2
    - [X] Loops on primitive
    - [X] Loops on complex objects
    - [ ] Loops without specific syntax
- [ ] Nice documentation