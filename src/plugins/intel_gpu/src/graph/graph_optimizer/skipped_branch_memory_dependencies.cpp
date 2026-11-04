// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_node.h"
#include "program_helpers.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include <algorithm>
#include <list>

using namespace cldnn;

void skipped_branch_memory_dependencies::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::SkippedBranchMemoryDependencies");
    // Primitive A can't use primitive B buffer if processing_num(B) < processing_num(A) and for any usr - the user of B
    // processing_num(usr) > processing_num(A) Otherwise it could override data that has to be used in the future.
    auto& processing_order = p.get_processing_order();
    auto itrB = processing_order.begin();
    while (itrB != processing_order.end()) {
        const auto& nodeB = *itrB;
        auto itrA = ++itrB;
        if (!nodeB->may_use_mempool()) {
            continue;
        }
        if (nodeB->get_users().empty()) {
            continue;
        }

        // find the last reader of B's buffer in processing order. A user that writes in place into B's buffer
        // (onednn sum post-op) keeps the buffer alive until the last reader of its own output.
        const auto readers = onednn_eltwise_fusing_helpers::get_buffer_readers(*nodeB);
        auto* lastUsr = *std::max_element(readers.begin(), readers.end(), [&](program_node* a, program_node* b) {
            return processing_order.get_processing_number(a) < processing_order.get_processing_number(b);
        });

        // mark all nodes in between B and lastUsr of B as forbidden to share buffer with B
        while (itrA != processing_order.get_processing_iterator(*lastUsr) && itrA != processing_order.end()) {
            const auto& nodeA = *itrA;
            itrA++;
            add_memory_dependency(nodeA, nodeB);
            add_memory_dependency(nodeB, nodeA);
        }
        // A last reader reached through an in-place user does not read B itself, so the input/output
        // restriction of basic_memory_dependencies does not keep it off B's buffer.
        const auto& lastUsrDeps = lastUsr->get_dependencies();
        if (std::none_of(lastUsrDeps.begin(), lastUsrDeps.end(), [&](const auto& dep) { return dep.first == nodeB; })) {
            add_memory_dependency(lastUsr, nodeB);
            add_memory_dependency(nodeB, lastUsr);
        }
    }
}
