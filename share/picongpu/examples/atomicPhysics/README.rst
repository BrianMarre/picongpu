atomicPhysics: atomic Physics example for PIConGPU based on the Empty example by Axel Huebl
============================

.. sectionauthor:: Axel Huebl <a.huebl (at) hzdr.de> and Brian Marre <b.marre (at) hzdr.de>
This is a minimum spec-exmaple for testing the still experimental atomic physics brnach of picongpu.
It uses mostly default algorithms [BirdsallLangdon]_ [HockneyEastwood]_ , except for a few changes listed below, to reduce computation time.
 - reduced super cell size, only 2x2x2 cells form a super cell
 - reduced number of particels overall, only 1 macro-ion and 1 macro-electron per super cell

Use this as a starting point for your own atomic physics picongpu simulations.

References
----------

.. [BirdsallLangdon]
        C.K. Birdsall, A.B. Langdon.
        *Plasma Physics via Computer Simulation*,
        McGraw-Hill (1985),
        ISBN 0-07-005371-5

.. [HockneyEastwood]
        R.W. Hockney, J.W. Eastwood.
        *Computer Simulation Using Particles*,
        CRC Press (1988),
        ISBN 0-85274-392-0
