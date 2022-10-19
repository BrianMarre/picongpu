/* Copyright 2022 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/particles/atomicPhysics2/atomicData/ChargeStateData.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/ChargeStateOrgaData.hpp"

#include "picongpu/particles/atomicPhysics2/atomicData/AtomicStateData.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/TransitionSelectionData.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicStateOrgaData_BoundBound.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicStateOrgaData_BoundFree.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicStateOrgaData_Autonomous.hpp"

#include "picongpu/particles/atomicPhysics2/atomicData/AutonomousTransitionData.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/BoundBoundTransitionData.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/BoundFreeTransitionData.hpp"


#include "picongpu/particles/atomicPhysics2/atomicData/TransitionSelectionData.hpp"

/** @file gathering of all files implementing storage classes of atomic Data
 *
 * The atomicPhysics step relies on a model of atomic states and transitions for each
 * atomicPhysics ion species.
 * These model's parameters are provided by the user as a .txt file of specified format
 * at runtime, due to license requirements.
 *
 *  PIConGPU itself only includes charge state data, for ADK-, Thomas-Fermi- and BSI-ionization.
 *  All other atomic state data is kept separate from PIConGPU itself.
 *
 * This file is read at the start of the simulation and stored in several objects for
 *  later use.
 *
 * Always two different classes handle each sub set of atomicPhysics data:
 * - a data class ... implements
 *                      * reading of the atomicData input file
 *                      * export to the DataBox for device side(GPU) use
 *                      * host side storage of atomicData
 * - a DataBox class ... device side(GPU) storage and access to atomicData
 *
 * For some data sets also an separate orga-data set exists. This describes the Structure
 *  of the pure value data for faster lookups.
 */