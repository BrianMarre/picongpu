/* Copyright 2022-2024 Brian Marre
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

//! @file collection of all IPD models for easier include

#pragma once

#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/fieldBarrierSupression/Model_BSI.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/noIPD/Model_NoIPD_Matter.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stewartPyatt/Model_StewartPyatt.hpp"
