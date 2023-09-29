#import "template.typ": *
#show: ieee.with(
  title: "Path Tracer na platformě CUDA",
  abstract: [Poznámky k projektu na PGRF3. Velice laické. Nemusí být zcela korektní. Work-In-Progress.],
  authors: (
    (
      name: "Tomáš Král",
    ),
  ),
  paper-size: "a4",
  index-terms: (),
  bibliography-file: "refs.bib",
)

= Programovací model GPU

== Grafický hardware
Základní vlastnost hardwaru od firmy Nvidia je, že výpočty jsou vždy prováděny ve skupině 32 threadů, která se nazývá _warp_. Všechny thready ve warpu běží paralelně. Problém může nastat ve chvíli, kdy některé thready potřebují vykonat jiný kód než ostatní thready. Taková situace může nastat například když jedna skupina threadů splňuje podmínku v konstruktu `if-else` a druhá nikoliv. Tato situace se ozančuje pojmem _divergence_.

Divergence je na hardwaru řešena tím, že některé thready jsou _zamaskovány_, tzn. po dobu vykonávání divergentní části kódu jsou vypnuty, viz. @divergence_img:

#figure(
  image("images/divergence.png"),
  caption: [Vizualizace maskování threadů při divergenci #cite("nvidia_volta").],
  kind: "obr",
  supplement: [Obr.],
) <divergence_img>

Divergence má pochopitelně negativní vliv na rychlost výpočtu - čím více budou jednotlivé thready divergovat, tím nižší bude úroveň paralelizace. Pro maximálně efektivní využití hardwaru je tedy nutné implementovat algoritmy tak, aby běh programu pokud možno co nejméně divergoval.

== Paralelizace Path Traceru

Path-Tracing je možné paralelizovat několika způsoby. Asi nejjednodušší by bylo rozdělit výpočet na úrovni pixelů - co jeden pixel, to jeden thread. Jasnou nevýhodou takového postupu je vysoká divergence. Známou vlastností algoritmů path-tracingu totiž je, že množství výpočtů se pro různé pixely může masivně lišit. To je způsobené například odlišnou složitostí scény v různých segmentech obrazu.

Lepším způsobem by bylo paralelizovat výpočet jednotlivých vzorků jednoho pixelu. Pro typické použití Path-Tracingu je totiž běžné pro každý pixel počítat stovky až tisíce vzorků. Touto cestou je možné do jisté míry snížit množství divergence vycházející z rozdílných výpočetních nároků pro různé pixely. Nicméně zůstává divergence způsobená např. výpočtem různých BRDF, materiálů, průsečíků s různými geometrickými útvary atd.

Řešením problému paralelizace je tzv. _Wavefront_ algoritmus #cite("10.1145/2492045.2492060"), který vychází z tzv. _streaming path-tracing_ postupu #cite("van_antwerpen_unbiased_2011").

= Implementace
== Konfigurace systému a požadavky
Implementace byla provedena na platformě CUDA 12.2. Byl použit standard C++23 na kompilátoru GCC 13.2.1. Program byl testován na grafické kartě Nvidia GeForce MX 550M na OS Linux s verzí kernelu 6.5.4.

Implementace používá CUDA funkcionalitu jménem _Unified Memory_ #cite("cuda_programming_guide"), která vyžaduje GPU s architekturou SM 3.0 nebo vyšší (řada Kepler a novější). Některé pokročilé funkce _Unified Memory_ jsou dostupné pouze na OS Linux, ty ale *doufám* nebyly použity.

Bylo použito několik externích knihoven: fmt, GLM. Tyto knihovny jsou do projektu zakomponovány pomocí package manageru vcpkg a buildovacího systému CMake.
